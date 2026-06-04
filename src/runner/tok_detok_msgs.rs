// Wire protocol for engine ↔ tok_detok_worker IPC.
//
// Format on the wire: [len: u32 LE][kind: u8][bincode(payload): len bytes]

use serde::{Deserialize, Serialize};

use crate::utils::downloader::ModelPaths;

#[derive(Copy, Clone, Debug)]
#[repr(u8)]
pub enum MsgKind {
    TokDetokInit = 1,
    Tokenize = 2,
    TokenizeResp = 3,
    Detokenize = 4,
    DetokenizeResp = 5,
    Error = 99,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TokDetokInit {
    pub model_paths: ModelPaths,
    pub is_gguf: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TokenizeReq {
    pub prompt: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TokenizeResp {
    pub token_ids: Vec<u32>,
    pub prompt_len: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DetokenizeReq {
    pub token_ids: Vec<u32>,
    pub skip_special_tokens: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DetokenizeResp {
    pub text: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WorkerError {
    pub msg: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    // Wire-format guard: discriminants are part of the IPC contract.
    #[test]
    fn msg_kind_discriminants_stable() {
        assert_eq!(MsgKind::TokDetokInit as u8, 1);
        assert_eq!(MsgKind::Tokenize as u8, 2);
        assert_eq!(MsgKind::TokenizeResp as u8, 3);
        assert_eq!(MsgKind::Detokenize as u8, 4);
        assert_eq!(MsgKind::DetokenizeResp as u8, 5);
        assert_eq!(MsgKind::Error as u8, 99);
    }

    #[test]
    fn tokenize_req_roundtrip() {
        let req = TokenizeReq {
            prompt: "hello world".to_string(),
        };
        let bytes = bincode::serialize(&req).unwrap();
        let back: TokenizeReq = bincode::deserialize(&bytes).unwrap();
        assert_eq!(back.prompt, req.prompt);
    }

    #[test]
    fn tokenize_resp_roundtrip() {
        let resp = TokenizeResp {
            token_ids: vec![1, 2, 3, 42, 65535],
            prompt_len: 5,
        };
        let bytes = bincode::serialize(&resp).unwrap();
        let back: TokenizeResp = bincode::deserialize(&bytes).unwrap();
        assert_eq!(back.token_ids, resp.token_ids);
        assert_eq!(back.prompt_len, resp.prompt_len);
    }

    #[test]
    fn detokenize_req_roundtrip() {
        let req = DetokenizeReq {
            token_ids: vec![100, 200, 300],
            skip_special_tokens: true,
        };
        let bytes = bincode::serialize(&req).unwrap();
        let back: DetokenizeReq = bincode::deserialize(&bytes).unwrap();
        assert_eq!(back.token_ids, req.token_ids);
        assert_eq!(back.skip_special_tokens, req.skip_special_tokens);
    }

    #[test]
    fn detokenize_resp_roundtrip() {
        let resp = DetokenizeResp {
            text: "hello world".to_string(),
        };
        let bytes = bincode::serialize(&resp).unwrap();
        let back: DetokenizeResp = bincode::deserialize(&bytes).unwrap();
        assert_eq!(back.text, resp.text);
    }
}
