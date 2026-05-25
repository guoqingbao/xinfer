use candle_core::Result;
use clap::Parser;
use colored::Colorize;
use reedline::{DefaultPrompt, DefaultPromptSegment, Reedline, Signal};
use serde_json;
use std::sync::Arc;
use tool_parser::ParserFactory;
use xinfer::core::engine::StreamItem;
use xinfer::core::engine::GLOBAL_RT;
use xinfer::core::{engine::LLMEngine, GenerationOutput};
use xinfer::log_error;
use xinfer::server::run_server;
use xinfer::server::Args;
use xinfer::transfer::{PdConfig, PdMethod, PdRole};
use xinfer::utils::chat_template::Message;
use xinfer::utils::config::GenerationConfig;
use xinfer::utils::config::{EngineConfig, SamplingParams};
use xinfer::utils::get_dtype;

#[tokio::main]
async fn main() -> Result<()> {
    // When invoked as `xinfer runner --sock ... --uuid ...`, run the GPU worker subprocess.
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && args[1] == "runner" {
        let runner_args: Vec<String> = std::iter::once(args[0].clone())
            .chain(args[2..].iter().cloned())
            .collect();
        xinfer::runner::run_runner_process(runner_args)
            .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;
        return Ok(());
    }
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    let args = Args::parse();

    if args.model_id.is_none() && args.weight_path.is_none() && args.weight_file.is_none() {
        candle_core::bail!("Must provide model_id or weight_path or weight_file!");
    }

    if let Some(ref enforced) = args.enforce_parser {
        let parsers = ParserFactory::new().list_parsers();
        if !parsers.contains(enforced) {
            candle_core::bail!(
                "Invalid enforce-parser '{}'. Valid parsers: {}",
                enforced,
                parsers.join(", ")
            );
        }
    }

    let dtype = get_dtype(args.dtype);

    let (max_num_seqs, interactive) = if args.batch.is_some() {
        tracing::warn!("max_num_seqs is ignored in batch performance test.");
        if args.interactive {
            tracing::warn!("interactive mode is ignored in batch performance test.");
        }
        (args.batch.unwrap(), false)
    } else {
        (args.max_num_seqs, args.interactive)
    };

    assert!(
        !(interactive && (args.server || args.ui_server || args.pd_server)),
        "You selected both interactive and server mode, which is not valid!"
    );

    let max_model_len = {
        // If not explicitly set, let the allocator auto-decide a fit based on
        // post-load free memory. Interactive mode previously forced a large
        // default here, which incorrectly turned planning into a hard
        // "must fit this exact context length" requirement.
        assert!(
            args.max_model_len == None || args.kv_fraction == None,
            "You provided both max_model_len and kv_fraction!"
        );
        args.max_model_len
    };

    let server = args.server
        || args.ui_server
        || args.pd_server
        || (!interactive && args.prompts.is_none() && args.batch.is_none());

    let prompts = match (&args.prompts, interactive) {
        (Some(prompts), false) => prompts.clone(),
        (None, false) => {
            vec![]
        }
        (Some(_), true) => {
            tracing::warn!("Interactive mode does not support predefined prompts, these prompts will be ignored.");
            vec![]
        }
        (None, true) => {
            tracing::warn!("Enter interactive mode.");
            vec![]
        }
    };

    let prompts = if args.batch.is_some() {
        let prompts = if prompts.len() > 0 {
            vec![prompts.clone()[0].clone()]
        } else {
            vec!["Please talk about China in more details.".to_string()]
        };
        let repeated: Vec<String> = (0..args.batch.unwrap())
            .flat_map(|_| prompts.iter().cloned())
            .collect();
        repeated
    } else {
        prompts
    };

    let generation_cfg = if (args.temperature.is_some()
        && (args.top_k.is_some() && args.top_p.is_some()))
        || args.frequency_penalty.is_some()
        || args.presence_penalty.is_some()
    {
        Some(GenerationConfig {
            temperature: args.temperature,
            top_p: args.top_p,
            top_k: args.top_k,
            frequency_penalty: args.frequency_penalty,
            presence_penalty: args.presence_penalty,
            bos_token_id: None,
            eos_token_id: None,
        })
    } else {
        None
    };

    if args.pd_server && args.pd_client {
        candle_core::bail!("This program can only be served as PD server or PD client, not both!");
    }

    if args.pd_server && args.ui_server {
        candle_core::bail!("PD Server cannot run with UI Server enabled!");
    }

    #[cfg(not(feature = "cuda"))]
    if (args.pd_server || args.pd_client) && args.pd_url.is_none() {
        candle_core::bail!("Non-CUDA platform does not support LocalIPC, please provide pd-url (e.g., 0.0.0.0:8100)!");
    }

    let pd_config = if args.pd_server || args.pd_client {
        let pd_role = if args.pd_server {
            PdRole::Server
        } else {
            PdRole::Client
        };
        // RemoteTcp for http:// or tcp:// URLs, LocalIpc for file:// or None
        let pd_method = match &args.pd_url {
            Some(url) if url.starts_with("tcp:") || url.starts_with("http:") => PdMethod::RemoteTcp,
            _ => PdMethod::LocalIpc,
        };
        Some(PdConfig {
            role: pd_role,
            method: pd_method,
            url: args.pd_url,
        })
    } else {
        None
    };

    let prefix_cache = !args.disable_prefix_cache;
    let tool_prompt_template = if let Some(ref path) = args.tool_prompt {
        let content = std::fs::read_to_string(path).map_err(candle_core::Error::wrap)?;
        let json: serde_json::Value =
            serde_json::from_str(&content).map_err(candle_core::Error::wrap)?;
        json.get("tool_prompt")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    } else {
        None
    };

    if let Some(ref dtype_str) = args.kvcache_dtype {
        use xinfer::utils::config::KvCacheDtype;
        KvCacheDtype::from_str_opt(dtype_str).unwrap_or_else(|| {
            panic!(
                "Invalid --kvcache-dtype value: {}. Use auto/fp8/turbo8/turbo4/turbo3.",
                dtype_str
            )
        });
    }
    let econfig = EngineConfig::new(
        args.model_id,
        args.weight_path,
        args.weight_file,
        args.hf_token,
        args.hf_token_path,
        args.enforce_parser.clone(),
        Some(std::cmp::max(max_num_seqs, prompts.len())),
        None,
        max_model_len,
        Some(args.max_tokens),
        args.isq.clone(),
        Some(1),
        args.device_ids.clone(),
        generation_cfg,
        args.seed,
        !prefix_cache,
        args.prefix_cache_max_tokens,
        args.kvcache_dtype.clone(),
        Some(server || !interactive),
        args.cpu_mem_fold,
        args.kv_fraction,
        args.mamba_fraction,
        pd_config,
        args.mcp_command.clone(),
        args.mcp_config.clone(),
        args.mcp_args.clone(),
        tool_prompt_template,
        None, // pd_server_prefix_cache_ratio
        None, // pd_client_prefix_cache_ratio
        args.yarn_scaling_factor,
        args.disable_reasoning,
        args.disable_cuda_graph,
        Some(args.prefill_chunk_size),
    );

    let server_port = if server {
        let port = args
            .port
            .unwrap_or(if args.pd_server { 7000 } else { 8000 });
        xinfer::utils::ensure_port_free("0.0.0.0", port as u16);
        Some(port)
    } else {
        None
    };

    let engine = LLMEngine::new(&econfig, dtype)?;
    if let Some(port) = server_port {
        run_server(engine.clone(), econfig.clone(), port, args.ui_server).await?;
        return Ok(());
    }

    if !interactive && prompts.is_empty() {
        eprintln!(
            "{}",
            String::from("⛔️ No prompts provided for completion. Use --i for interactive mode or --server for server mode.").red()
        );
        return Ok(());
    }

    let mut params = Vec::new();
    let mut message_list = Vec::new();
    // let mut rng = rand::rng();
    if !interactive && prompts.len() > 0 {
        if prompts.len() > 1 {
            tracing::warn!("Live output muted for more than one prompt!\n");
        }
        for prompt in prompts.iter() {
            let msg = Message::new("user".to_string(), prompt.clone(), 0);
            let param = SamplingParams::new_with_max_tokens(args.max_tokens);
            message_list.push(vec![msg]);
            params.push(param);
        }
    } else {
        params.push(SamplingParams::new_with_max_tokens(args.max_tokens));
    }

    let (resolved_max_model_len, resolved_max_num_seqs) = {
        let e = engine.read();
        let (_, _, mml) = e.get_model_info();
        (mml.unwrap_or(32768usize), e.econfig.max_num_seqs)
    };

    if args.max_tokens > resolved_max_model_len {
        log_error!(
            "Requested max_tokens {} larger than max_model_len {}",
            args.max_tokens,
            resolved_max_model_len
        );
    }

    tracing::info!(
        "Resolved max_model_len: {}, max_num_seqs: {}",
        resolved_max_model_len,
        resolved_max_num_seqs
    );

    let total_available_tokens: i64 = resolved_max_num_seqs as i64 * resolved_max_model_len as i64;
    let mut chat_context_left = total_available_tokens;
    let mut line_editor = Reedline::create();
    let mut prompt = DefaultPrompt {
        left_prompt: DefaultPromptSegment::WorkingDirectory,
        right_prompt: DefaultPromptSegment::Basic(format!(
            "Tokens left: {} (full)",
            chat_context_left
        )),
    };

    let request_params = params[0].clone();
    let mut chat_history = Vec::<Message>::new();
    loop {
        if interactive {
            if chat_history.is_empty() {
                print!("🤖✨ Enter a new prompt (Press Ctrl+C to exit):");
            } else {
                print!("🤖✨ Enter another prompt to continue current chat (Press Ctrl+C to start a new chat):\n");
            }
            let sig = line_editor.read_line(&prompt);
            if chat_context_left < 0 {
                tracing::error!("No tokens left, press Ctrl+C to start a new chat session!");
                chat_context_left = 0;
                continue;
            }
            match sig {
                Ok(Signal::Success(buffer)) => {
                    let trimmed = buffer.trim();
                    if !trimmed.is_empty() {
                        let msg = Message::new("user".to_string(), trimmed.to_string(), 0);
                        chat_history.push(msg.clone());
                    } else {
                        print!("\n No prompt was given.");
                        continue;
                    }
                }
                Ok(Signal::CtrlD) | Ok(Signal::CtrlC) => {
                    if chat_history.is_empty() {
                        print!("\n👋 Exiting.");
                        std::process::exit(0); // Ctrl+C to exit
                    } else {
                        print!("\n🌀 Chat history cleared. Start a new conversation.\n");
                        chat_history.clear(); //start a new chat
                        if prefix_cache {
                            let e = engine.read();
                            chat_context_left =
                                total_available_tokens - e.get_num_cached_tokens() as i64;
                        } else {
                            chat_context_left = total_available_tokens;
                        }
                        prompt.right_prompt = DefaultPromptSegment::Basic(format!(
                            "Tokens left: {} (full)",
                            chat_context_left
                        ));
                        continue;
                    }
                }
                _ => {}
            }
        }

        let mut outputs = {
            if interactive {
                let (seq_id, prompt_length, _prefilled_reasoning_end, stream) = {
                    let mut e = engine.write();
                    match e.generate_stream(
                        &request_params,
                        &chat_history,
                        None,
                        &Vec::new(),
                        &None,
                    ) {
                        Ok((seq_id, prompt_length, prefilled_reasoning_end, stream)) => {
                            (seq_id, prompt_length, prefilled_reasoning_end, stream)
                        }
                        Err(e) => {
                            tracing::error!("Session unexpectedly ended because: {:?}", e);
                            print!("\n🌀 Chat history cleared. Start a new conversation.\n");
                            chat_history.clear(); //start a new chat
                            prompt.right_prompt = DefaultPromptSegment::Basic(format!(
                                "Tokens left: {} (full)",
                                total_available_tokens
                            ));
                            continue;
                        }
                    }
                };
                let handle: tokio::task::JoinHandle<(usize, usize, usize, usize, String)> =
                    GLOBAL_RT.spawn(async move {
                        let mut tickets: (usize, usize, usize, usize) = (0, 0, 0, 0);
                        let mut decode_output = "".to_string();
                        let mut rx = stream;
                        while let Some(item) = rx.recv().await {
                            match item {
                                StreamItem::Token(t, _token_id) => {
                                    decode_output += &t.to_string();
                                    print!("{}", t);
                                    use std::io::Write;
                                    let _ = std::io::stdout().flush();
                                }
                                StreamItem::TokenID(_) | StreamItem::Completion(_) => {
                                    break;
                                }
                                StreamItem::Done((
                                    prompt_start_time,
                                    decode_start_time,
                                    decode_finish_time,
                                    length,
                                    _stop_sequence,
                                )) => {
                                    tickets = (
                                        prompt_start_time,
                                        decode_start_time,
                                        decode_finish_time,
                                        length,
                                    );
                                    eprintln!(
                                        "{}",
                                        String::from("\r\nGeneration completed!").yellow()
                                    );
                                    break;
                                }
                                StreamItem::Error(e) => eprintln!("Error: {}", e),
                            }
                        }
                        (tickets.0, tickets.1, tickets.2, tickets.3, decode_output)
                    });
                let (
                    prompt_start_time,
                    decode_start_time,
                    decode_finish_time,
                    decoded_length,
                    decode_output,
                ) = handle.await.map_err(candle_core::Error::wrap)?;
                if prefix_cache {
                    let e = engine.read();
                    chat_context_left = total_available_tokens - e.get_num_cached_tokens() as i64;
                } else {
                    chat_context_left =
                        total_available_tokens - prompt_length as i64 - decoded_length as i64;
                }
                prompt.right_prompt =
                    DefaultPromptSegment::Basic(format!("Tokens left: {}", chat_context_left));
                vec![GenerationOutput {
                    seq_id,
                    prompt_length,
                    prompt_start_time,
                    decode_start_time,
                    decode_finish_time,
                    decoded_length,
                    decode_output,
                    stop_sequence: None,
                }]
            } else {
                xinfer::log_warn!("Starting the inference...");

                let (receivers, tokenizer) = {
                    let mut e = engine.write();
                    (
                        e.generate_sync(&params, &message_list, None, &Vec::new(), &None)?,
                        Arc::new(e.tokenizer.clone()),
                    )
                };
                let results = LLMEngine::collect_sync_results(receivers, tokenizer, None).await?;
                // GenerationOutput is returned directly
                results
            }
        };

        outputs.sort_by_key(|o| o.seq_id);

        let mut all_decode_time_taken = 0f32;
        let mut prompt_time_taken = 0f32;
        let mut total_decoded_tokens = 0;
        let mut total_prompt_tokens = 0;

        for (
            i,
            GenerationOutput {
                seq_id,
                prompt_length,
                prompt_start_time,
                decode_start_time,
                decode_finish_time,
                decoded_length,
                decode_output,
                stop_sequence: _,
            },
        ) in outputs.iter().enumerate()
        {
            if !interactive && args.batch.is_none() {
                tracing::info!("[seq_id {}] 📚✨ Prompt {}: {}", seq_id, i + 1, prompts[i]);
                tracing::info!("[seq_id {}] 📄✨ Response: {}\n", seq_id, decode_output);
            }
            total_prompt_tokens += prompt_length;
            total_decoded_tokens += decoded_length;
            let duration_prompt = (decode_start_time - prompt_start_time) as f32 / 1000.0; //maximum time costs for prompt
            if duration_prompt > prompt_time_taken {
                prompt_time_taken = duration_prompt;
            }

            let duration = (decode_finish_time - decode_start_time) as f32 / 1000.0; //maximum time costs for decoding
            all_decode_time_taken += duration;

            if interactive {
                let msg = Message::new("assistant".to_string(), decode_output.to_string(), 0);
                chat_history.push(msg.clone());
            }
        }

        let decode_time_taken = all_decode_time_taken / outputs.len() as f32;
        xinfer::log_info!("--- Performance Metrics ---");

        tracing::info!(
            "⏱️ Prompt tokens: {} in {:.2}s ({:.2} tokens/s)",
            total_prompt_tokens,
            prompt_time_taken,
            total_prompt_tokens as f32 / prompt_time_taken,
        );
        tracing::info!(
            "⏱️ Decoded tokens: {} in {:.2}s ({:.2} tokens/s)",
            total_decoded_tokens,
            decode_time_taken,
            total_decoded_tokens as f32 / decode_time_taken,
        );

        if !interactive {
            break;
        }
    }

    Ok(())
}
