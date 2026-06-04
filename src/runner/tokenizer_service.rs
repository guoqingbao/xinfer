// TokenizerService — single dispatch enum for the engine's hot-path
// tokenize/detokenize calls. Mirrors the `RunnerType::{Thread, Process}`
// shape in `src/core/runner.rs`: keep every variant's complexity off the
// engine hot path, and let future variants (inference-firewall,
// grammar-aware, remote multi-model) plug in by adding a variant rather
// than threading another `Option<...>` through every call site.

use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use tokenizers::Tokenizer;

use crate::runner::tok_detok_msgs::{
    DetokenizeReq, DetokenizeResp, MsgKind, TokenizeReq, TokenizeResp,
};
use crate::runner::tok_detok_socket::TokDetokIpcPair;

#[derive(Clone)]
pub enum TokenizerService {
    /// In-process tokenizer. Default. Bit-identical to pre-PR behavior.
    Inline(Arc<Tokenizer>),
    /// Out-of-process worker via `interprocess::local_socket`. Opt-in via
    /// `XINFER_TOK_DETOK_WORKER=1`. `tokenizer` is still held so chat
    /// templates, `/tokenize`, samplers, and other non-hot-path consumers
    /// don't pay the IPC tax.
    Worker {
        ipc: TokDetokIpcPair,
        tokenizer: Arc<Tokenizer>,
    },
}

impl TokenizerService {
    /// Underlying tokenizer — always available, even in Worker mode.
    /// Non-hot-path consumers (chat template, `/tokenize` endpoint, stream
    /// decoder) hold the in-process tokenizer directly via this accessor.
    pub fn tokenizer(&self) -> &Arc<Tokenizer> {
        match self {
            Self::Inline(t) => t,
            Self::Worker { tokenizer, .. } => tokenizer,
        }
    }

    /// Hot-path encode. Returns `(token_ids, prompt_len)`.
    pub fn encode(&self, prompt: &str) -> Result<(Vec<u32>, usize)> {
        match self {
            Self::Inline(tok) => {
                let enc = tok
                    .encode_fast(prompt, true)
                    .map_err(|e| anyhow!("encode failed: {e:?}"))?;
                let ids: Vec<u32> = enc.get_ids().to_vec();
                let len = ids.len();
                Ok((ids, len))
            }
            Self::Worker { ipc, .. } => {
                let req = TokenizeReq {
                    prompt: prompt.to_string(),
                };
                let bytes = bincode::serialize(&req)?;
                ipc.tok.send(&bytes, MsgKind::Tokenize);
                let (kind, data) = ipc
                    .tok
                    .recv_blocking()?
                    .ok_or_else(|| anyhow!("tok_detok_worker tok socket closed"))?;
                if kind != MsgKind::TokenizeResp as u8 {
                    bail!("tok_detok_worker: unexpected tok kind={}", kind);
                }
                let resp: TokenizeResp = bincode::deserialize(&data)?;
                let len = resp.prompt_len;
                Ok((resp.token_ids, len))
            }
        }
    }

    /// Hot-path decode of a full token sequence (one-shot, non-streaming).
    /// Streaming decode uses `tokenizers::DecodeStream` directly against
    /// the underlying tokenizer — see [`TokenizerService::tokenizer`].
    pub fn decode(&self, tokens: &[u32], skip_special: bool) -> Result<String> {
        match self {
            Self::Inline(tok) => tok
                .decode(tokens, skip_special)
                .map_err(|e| anyhow!("decode failed: {e:?}")),
            Self::Worker { ipc, .. } => {
                let req = DetokenizeReq {
                    token_ids: tokens.to_vec(),
                    skip_special_tokens: skip_special,
                };
                let bytes = bincode::serialize(&req)?;
                ipc.det.send(&bytes, MsgKind::Detokenize);
                let (kind, data) = ipc
                    .det
                    .recv_blocking()?
                    .ok_or_else(|| anyhow!("tok_detok_worker det socket closed"))?;
                if kind != MsgKind::DetokenizeResp as u8 {
                    bail!("tok_detok_worker: unexpected det kind={}", kind);
                }
                let resp: DetokenizeResp = bincode::deserialize(&data)?;
                Ok(resp.text)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Inline encode/decode round-trip against a real tokenizer.
    // We don't test the Worker variant here because it would need a live
    // subprocess; that's covered by the integration smoke in the bench.
    #[test]
    fn inline_encode_decode_roundtrip() {
        // Build a trivially-small tokenizer for the test — bytelevel BPE
        // over ASCII is enough to exercise the round-trip plumbing.
        let json = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true},"post_processor":null,"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7,"i":8,"j":9,"k":10,"l":11,"m":12,"n":13,"o":14,"p":15,"q":16,"r":17,"s":18,"t":19,"u":20,"v":21,"w":22,"x":23,"y":24,"z":25,"Ġ":26},"merges":[]}}"#;
        let tok = Tokenizer::from_bytes(json.as_bytes()).expect("build tokenizer");
        let svc = TokenizerService::Inline(Arc::new(tok));
        let (ids, len) = svc.encode("hello").expect("encode");
        assert_eq!(len, ids.len());
        assert!(!ids.is_empty());
        let _text = svc.decode(&ids, true).expect("decode");
    }
}
