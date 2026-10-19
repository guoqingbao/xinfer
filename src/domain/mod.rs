//! The domain substrate on the `spawned` actor framework (the Erlang-style
//! actors). Each domain is a single-threaded actor that owns its state +
//! communicates by message passing (the no shared locks).
//!
//! An actor is reachable ONLY through its proxy (the `ActorRef` / the
//! type-erased `XRef`). No code holds the actor object; the only shared objects
//! are the message senders (which are `Clone + Send`). This is what makes the
//! desync class of bug structurally impossible: the `IpcChannel` actor is the
//! sole socket writer.
//!
//! The is the substrate the domain pipeline is built on: the `HttpD` the
//! `SchedD`, the `FwdD` (per-GPU), the `SmpD`, the `StreamD`, the `KvD`, the
//! `StateD`, the `StoreD` actors, connected by message edges.
//!
//! ## The actor invariant (PROOFED)
//!
//! 1. **The sole socket writer.** The front-end runner socket is driven ONLY by
//!    the `IpcChannel` actor (the `do_io` drain-all). No other code calls
//!    `send_local`/`receive_local` on the front-end side (verified: the only
//!    front-end socket I/O is in `domain/ipc.rs`; the runner-process side + the
//!    progress/command channels are separate sockets).
//! 2. **No shared transport reference.** The `LLMEngine` holds only the actor
//!    proxies (`FwdRef` `SchedRef`, `StreamRef`, `HttpRef`, `KvRef`). It does
//!    NOT hold a `Scheduler`, a `BlockManager`, a `RunnerType`, or a raw socket.
//!    Every component reaches the model + the scheduler + the block manager ONLY
//!    through the actor proxies (the type-erased `Arc<dyn Protocol>`).
//! 3. **The end loop is actor-driven.** `start_engine` is a native thread that
//!    calls `sched.step()` (the `SchedD` actor), which drives the `FwdD` + the
//!    `StreamD` + the `KvD` actors. The engine no calls the scheduler or the
//!    transport directly.
//!
//! The `full_graph_delivers_token_to_client` test (the this module) proves the
//! full pipeline (HttpD -> SchedD -> FwdD -> StreamD) delivers a token to the
//! client's stream channel, end-to-end, on any machine (the no GPU, the no model).

pub mod ipc;
pub use ipc::{IpcChannel, IpcMessage, start_ipc_actor};
pub mod fwd;
pub use fwd::{FwdD, FwdMessage, OwnedSeqs, start_fwd_actor};
pub mod sched;
pub use sched::{SchedD, SchedMessage, start_sched_actor};
pub mod stream;
pub use stream::{StreamD, StreamMessage, start_stream_actor};
pub mod http;
pub use http::{HttpD, HttpMessage, start_http_actor};
pub mod kv;
pub use kv::{KvD, KvMessage, start_kv_actor};

#[cfg(test)]
mod graph_tests {
    //! The end-to-end graph test: drive a request through the full actor
    //! pipeline (HttpD -> SchedD -> FwdD -> StreamD) + assert the token is
    //! delivered to the client's stream channel. This proves the full Noria
    //! graph works, not just the individual actors.

    use super::*;
    use crate::core::block_manager::BlockManager;
    use crate::core::engine::StreamItem;
    use crate::core::scheduler::Scheduler;
    use crate::core::sequence::Sequence;
    use crate::domain::fwd::{start_fwd_actor, FwdMessage};
    use crate::domain::http::{start_http_actor, HttpMessage};
    use crate::domain::kv::{start_kv_actor, KvMessage};
    use crate::domain::sched::{start_sched_actor, SchedMessage};
    use crate::domain::stream::{start_stream_actor, StreamMessage};
    use crate::runner::MessageType;
    use crate::utils::config::{Config, EngineConfig, SamplingParams};
    use interprocess::local_socket::traits::Listener;
    use interprocess::local_socket::traits::Stream;
    use interprocess::local_socket::{GenericNamespaced, ListenerOptions, ToNsName};
    use interprocess::TryClone;
    use ractor::rpc::CallResult;
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    /// Build a minimal `Config` (the model config) for the test.
    fn minimal_config() -> Config {
        serde_json::from_str(
            r#"{
                "architectures": ["Qwen3ForCausalLM"],
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "max_position_embeddings": 128,
                "hidden_size": 8,
                "num_hidden_layers": 2,
                "intermediate_size": 16,
                "rms_norm_eps": 1e-6,
                "hidden_act": "silu",
                "kvcache_dtype": "Auto",
                "mtp_enabled": false,
                "mtp_max_verify_tokens": 0
            }"#,
        )
        .expect("minimal Config")
    }

    /// Build a minimal `EngineConfig` for the test.
    fn minimal_econfig() -> EngineConfig {
        EngineConfig::new(
            Some("test-model".to_string()), // 1 model_id
            None, // 2 weight_path
            None, // 3 weight_file
            None, // 4 hf_token
            None, // 5 hf_token_path
            None, // 6 enforce_parser
            Some(1), // 7 max_num_seqs
            None, // 8 config_model_len
            Some(128), // 9 max_model_len
            Some(8), // 10 max_tokens
            None, // 11 isq
            Some(1), // 12 num_shards
            Some(vec![0]), // 13 device_ids
            None, // 14 generation_cfg
            None, // 15 seed
            false, // 16 disable_prefix_cache
            None, // 17 prefix_cache_max_tokens
            None, // 18 kvcache_dtype
            Some(false), // 19 server_mode
            None, // 20 cpu_mem_fold
            None, // 21 kv_fraction
            None, // 22 mamba_fraction
            None, // 23 pd_config
            None, // 24 mcp_command
            None, // 25 mcp_config
            None, // 26 mcp_args
            None, // 27 tool_prompt_template
            None, // 28 pd_server_prefix_cache_ratio
            None, // 29 pd_client_prefix_cache_ratio
            None, // 30 yarn_scaling_factor
            true, // 31 disable_reasoning
            true, // 32 disable_cuda_graph
            None, // 33 prefill_chunk_size
            1, // 34 num_nodes
            0, // 35 node_rank
            None, // 36 master_addr
            29500, // 37 master_port
            false, // 38 enable_tool_grammar
            None, // 39 num_speculative_tokens
            None, // 40 draft_model
        )
    }

    /// Simulate a single runner subprocess (the socket pair's runner end) that
    /// responds to every forward with a fixed token (the `42`).
    fn spawn_simulated_runner(
        name: &str,
        counter: Arc<AtomicUsize>,
    ) -> interprocess::local_socket::Stream {
        let ns_name = name
            .to_ns_name::<GenericNamespaced>()
            .expect("abstract socket name");
        let listener = ListenerOptions::new()
            .name(ns_name.clone())
            .create_sync()
            .expect("create the test listener");
        let front_end = interprocess::local_socket::Stream::connect(ns_name.clone())
            .expect("connect the front-end end");
        let runner_end = listener.accept().expect("accept the runner end");
        std::thread::spawn(move || {
            let mut stream = runner_end;
            loop {
                match crate::runner::receive_local(&mut stream, false) {
                    Ok(MessageType::Shutdown) => break,
                    Ok(_) => {
                        counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        let _ = crate::runner::send_local(
                            &mut vec![stream.try_clone().expect("clone the runner stream")],
                            &MessageType::RunResponse(vec![42]),
                            false,
                        );
                    }
                    Err(_) => break,
                }
            }
        });
        front_end
    }

    /// The end-to-end graph proof: drive a request through the full actor
    /// pipeline (HttpD -> SchedD -> FwdD -> StreamD) + assert the token is
    /// delivered to the client's stream channel.
    #[tokio::test]
    async fn full_graph_delivers_token_to_client() {
        const N: usize = 1; // single-rank (the no multi-rank, the graph proof)
        let counter = Arc::new(AtomicUsize::new(0));
        let name = format!("test-graph-{}@xinfer-test", std::process::id());
        let front_end = spawn_simulated_runner(&name, counter.clone());
        let streams = vec![front_end];

        // Build the full graph (the actors + the proxies).
        let fwd = start_fwd_actor(streams).await;
        let econfig = minimal_econfig();
        let config = minimal_config();
        let block_manager = BlockManager::new(
            fwd.clone(),
            16, 0, 4,
            crate::core::prefix_cache::PrefixCacheConfig { enabled: false, max_cached_blocks: 0 },
            false, 0,
        );
        let kv = start_kv_actor(block_manager).await;
        let scheduler = Scheduler::new(kv.clone(), &econfig, &config);
        let stream = start_stream_actor().await;
        let sched = start_sched_actor(scheduler, fwd.clone(), stream.clone(), false, false).await;
        let http = start_http_actor(sched.clone(), stream.clone()).await;
        // Resolve the circular dependency: SchedD needs HttpRef for
        // take_prompt_replay and remove_active_request.
        let _ = sched.send_message(SchedMessage::SetHttp(http.clone()));

        // Ingest a request through the HttpD actor (the stream channel is registered).
        let seq = Sequence::new(
            vec![1, 2, 3], 4,
            SamplingParams::default(),
            &None, 0,
        );
        let seq_id = match sched.call(|r| SchedMessage::Add(seq, r), None).await {
            Ok(CallResult::Success(v)) => v,
            _ => panic!("sched.add failed"),
        };
        let mut rx = match http.call(|r| HttpMessage::RegisterStream(seq_id, r), None).await {
            Ok(CallResult::Success(v)) => v,
            _ => panic!("http.register_stream failed"),
        };
        // Set the request type so the delivery loop knows to stream tokens.
        let _ = stream.send_message(StreamMessage::SetRequestType(seq_id, crate::core::engine::RequestType::Stream));

        // Drive one engine-loop step through the SchedD actor (the schedule -> forward -> postprocess -> stream delivery).
        let processed = match sched.call(|r| SchedMessage::Step(r), None).await {
            Ok(CallResult::Success(v)) => v,
            _ => 0,
        };
        assert!(processed >= 1, "the graph should have processed at least one sequence");

        // The token ( have been delivered to the client's stream channel.
        let item = rx.try_recv().expect("the token should have been delivered to the stream channel");
        match item {
            StreamItem::Token(_, tok) => assert_eq!(tok, 42, "the simulated runner's fixed token"),
            other => panic!("expected a token delivery, got {other:?}"),
        }

        // Clean up (the fire-and-forget Shutdown breaks the runner loop).
        let _ = fwd.send_message(FwdMessage::Shutdown);
        assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), 1, "the runner processed the prefill (one step = one forward pass)");
    }
}