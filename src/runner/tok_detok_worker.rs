// tok_detok_worker — runs tokenization and detokenization in a separate
// process, using two threads on two sockets so tokenize and detokenize
// requests don't head-of-line each other:
//
//     thread A ↔ XINFER_TOK_DETOK_SOCKET_TOK   (Tokenize / TokenizeResp / Error)
//     thread B ↔ XINFER_TOK_DETOK_SOCKET_DET   (Detokenize / DetokenizeResp / Error)
//
// Both threads share a single `Arc<Tokenizer>` (encode_fast / decode are
// thread-safe). Engine spawns this binary and binds two namespaced sockets;
// names are passed to the worker via env. Init is delivered on the tok
// socket and carries the model paths + GGUF flag.

use std::sync::Arc;
use std::thread;

use xinfer::log_info;
use xinfer::runner::tok_detok_msgs::{
    DetokenizeReq, DetokenizeResp, MsgKind, TokDetokInit, TokenizeReq, TokenizeResp, WorkerError,
};
use xinfer::runner::tok_detok_socket::TokDetokSocketClient;
use xinfer::utils::downloader::ModelPaths;
use xinfer::utils::gguf_helper::get_gguf_info;

use tokenizers::Tokenizer;

fn load_tokenizer(paths: &ModelPaths, is_gguf: bool) -> anyhow::Result<Tokenizer> {
    if is_gguf {
        let file = std::fs::File::open(&paths.get_weight_filenames()[0])?;
        let mut readers = vec![file];
        let mut readers = readers.iter_mut().collect::<Vec<_>>();
        let content = xinfer::utils::gguf_helper::Content::from_readers(&mut readers)
            .map_err(|_| anyhow::anyhow!("Unable to read GGUF"))?;
        let info = get_gguf_info(&content)?;
        Ok(info.tokenizer)
    } else {
        let tokenizer_file = paths.get_tokenizer_filename();
        let tok = Tokenizer::from_file(&tokenizer_file)
            .map_err(|e| anyhow::anyhow!("load tokenizer: {:?}", e))?;
        Ok(tok)
    }
}

fn run_tok_loop(client: TokDetokSocketClient, tokenizer: Arc<Tokenizer>) -> anyhow::Result<()> {
    loop {
        // `None` = engine closed its half of the socket on shutdown; exit cleanly.
        let (kind, data) = match client.recv_blocking()? {
            Some(frame) => frame,
            None => return Ok(()),
        };
        if kind != MsgKind::Tokenize as u8 {
            continue;
        }
        let req: TokenizeReq = bincode::deserialize(&data)?;
        match tokenizer.encode_fast(req.prompt.as_str(), true) {
            Ok(tokens) => {
                let ids: Vec<u32> = tokens.get_ids().iter().copied().collect();
                let resp = TokenizeResp {
                    token_ids: ids,
                    prompt_len: tokens.len(),
                };
                let bytes = bincode::serialize(&resp)?;
                client.send(&bytes, MsgKind::TokenizeResp);
            }
            Err(e) => {
                let bytes = bincode::serialize(&WorkerError {
                    msg: format!("tokenize error: {e:?}"),
                })?;
                client.send(&bytes, MsgKind::Error);
            }
        }
    }
}

fn run_det_loop(client: TokDetokSocketClient, tokenizer: Arc<Tokenizer>) -> anyhow::Result<()> {
    loop {
        // `None` = engine closed its half of the socket on shutdown; exit cleanly.
        let (kind, data) = match client.recv_blocking()? {
            Some(frame) => frame,
            None => return Ok(()),
        };
        if kind != MsgKind::Detokenize as u8 {
            continue;
        }
        let req: DetokenizeReq = bincode::deserialize(&data)?;
        match tokenizer.decode(&req.token_ids, req.skip_special_tokens) {
            Ok(text) => {
                let resp = DetokenizeResp { text };
                let bytes = bincode::serialize(&resp)?;
                client.send(&bytes, MsgKind::DetokenizeResp);
            }
            Err(e) => {
                let bytes = bincode::serialize(&WorkerError {
                    msg: format!("detokenize error: {e:?}"),
                })?;
                client.send(&bytes, MsgKind::Error);
            }
        }
    }
}

fn main() -> anyhow::Result<()> {
    log_info!("tok_detok_worker starting");
    let tok_sock = std::env::var("XINFER_TOK_DETOK_SOCKET_TOK").expect("XINFER_TOK_DETOK_SOCKET_TOK missing");
    let det_sock = std::env::var("XINFER_TOK_DETOK_SOCKET_DET").expect("XINFER_TOK_DETOK_SOCKET_DET missing");

    log_info!(
        "tok_detok_worker connecting tok={} det={}",
        tok_sock,
        det_sock
    );
    let tok_client = TokDetokSocketClient::connect(&tok_sock)?;
    let det_client = TokDetokSocketClient::connect(&det_sock)?;

    // Init arrives on the tok socket; carries shared config for both threads.
    let (kind, data) = tok_client
        .recv_blocking()?
        .ok_or_else(|| anyhow::anyhow!("engine closed tok socket before sending TokDetokInit"))?;
    if kind != MsgKind::TokDetokInit as u8 {
        anyhow::bail!("Expected TokDetokInit on tok socket, got kind={}", kind);
    }
    let init: TokDetokInit = bincode::deserialize(&data)?;
    let tokenizer = Arc::new(load_tokenizer(&init.model_paths, init.is_gguf)?);
    log_info!(
        "tok_detok_worker tokenizer loaded (is_gguf={}); spawning two service threads",
        init.is_gguf
    );

    let tokenizer_for_tok = Arc::clone(&tokenizer);
    let tok_handle = thread::spawn(move || {
        if let Err(e) = run_tok_loop(tok_client, tokenizer_for_tok) {
            eprintln!("tok_detok_worker: tok loop exited: {e:?}");
        }
    });
    let tokenizer_for_det = Arc::clone(&tokenizer);
    let det_handle = thread::spawn(move || {
        if let Err(e) = run_det_loop(det_client, tokenizer_for_det) {
            eprintln!("tok_detok_worker: det loop exited: {e:?}");
        }
    });

    let _ = tok_handle.join();
    let _ = det_handle.join();
    Ok(())
}
