//! The persistent inference-state store.
//!
//! A clean, contract-gated interface for persisting the inference state (the KV
//! cache + the GDN/mamba recurrent state + the prefix cache) to a Rust-native
//! backend: the filesystem (the CPU bounce-buffer path) or raw NVMe (the
//! GPUDirect-Storage zero-copy DMA path). Optional compression (zstd) and
//! authenticated encryption (AES-256-GCM, gated on XINFER_STATE_KEY) are applied
//! to the serialized state before it touches the backend.
//!
//! The store is the persistence half of the token-flow contract: state is
//! written only at a settled boundary (the gate has committed the tokens, the
//! KV/GDN are consistent), and restored only into a fresh engine that re-settles
//! before the first mask read.

use crate::utils::config::KvCacheDtype;
use aes_gcm::Aes256Gcm;
use aes_gcm::Nonce;
use aes_gcm::aead::Aead;
use aes_gcm::aead::KeyInit;
use std::path::{Path, PathBuf};

/// The persistent inference state: the KV cache, the GDN/mamba recurrent state,
/// and the prefix cache. This is the unit of persistence (a snapshot of one
/// sequence's or one instance's inference state).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct InferenceState {
    /// Per-layer KV, serialized as CPU byte blobs (D2H'd before save).
    pub kv: Vec<SerializedKvLayer>,
    /// GDN/mamba recurrent state (conv + ssm), empty for non-hybrid models.
    pub gdn: Vec<SerializedGdnState>,
    /// The prefix-cache hash (hash -> block)) so a restore can re-seed it.
    pub prefix_cache: SerializedPrefixCache,
    /// The KV dtype the tensors were stored in (for H2D re-materialization).
    pub kv_dtype: String,
}

/// One attention layer's K and V, flattened to CPU bytes.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SerializedKvLayer {
    pub k_bytes: Vec<u8>,
    pub v_bytes: Vec<u8>,
    /// The logical (num_blocks, num_heads, block_size, head_dim) shape.
    pub k_shape: Vec<usize>,
    pub v_shape: Vec<usize>,
}

/// One GDN/mamba layer's recurrent state, flattened to CPU bytes.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SerializedGdnState {
    pub conv_bytes: Vec<u8>,
    pub recurrent_bytes: Vec<u8>,
    pub conv_shape: Vec<usize>,
    pub recurrent_shape: Vec<usize>,
}

/// The prefix-cache mapping (hash -> block ids) for re-seeding on restore.
#[derive(Debug, Default, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SerializedPrefixCache {
    pub entries: Vec<(u64, Vec<usize>)>,
    pub block_size: usize,
}

/// The clean-agnostic persistent-state interface. Implementations: the
/// filesystem (CPU bounce buffer) and raw NVMe (GPUDirect-Storage zero-copy).
pub trait StateStore: Send + Sync {
    /// Persist `state` under `key`. The state is serialized, optionally
    /// compressed, optionally encrypted, then written to the backend.
    fn save(&self, key: &str, state: &InferenceState) -> Result<(), StateStoreError>;
    /// Load + deserialize the state under `key` (decrypt + decompress + rmp).
    fn load(&self, key: &str) -> Result<InferenceState, StateStoreError>;
    /// Delete the state under `key` (a no-op if absent).
    fn delete(&self, key: &str) -> Result<(), StateStoreError>;
    /// List the persisted keys.
    fn list(&self) -> Result<Vec<String>, StateStoreError>;
}

#[derive(Debug, thiserror::Error)]
pub enum StateStoreError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serialize(#[from] rmp_serde::encode::Error),
    #[error("deserialization error: {0}")]
    Deserialize(#[from] rmp_serde::decode::Error),
    #[error("compression error: {0}")]
    Compress(String),
    #[error("encryption error: {0}")]
    Crypto(String),
    #[error("key not found: {0}")]
    NotFound(String),
}

/// The filesystem backend (the CPU bounce-buffer path). The state is serialized
/// (rmp), optionally compressed (zstd), optionally encrypted (AES-256-GCM, the
/// XINFER_STATE_KEY), and written to `<root>/<key>.state`.
pub struct FsStateStore {
    root: PathBuf,
    compress: bool,
    /// The AES-256-GCM key (the XINFER_STATE_KEY, 32 bytes). None = no encryption.
    key: Option<Aes256Gcm>,
}

impl FsStateStore {
    pub fn new(root: impl AsRef<Path>, compress: bool, key: Option<Aes256Gcm>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
            compress,
            key,
        }
    }

    /// Build a store from the environment: compression from XINFER_STATE_COMPRESS,
    /// encryption from XINFER_STATE_KEY (a 32-byte hex or raw key).
    pub fn from_env(root: impl AsRef<Path>) -> std::io::Result<Self> {
        let compress = crate::utils::env::state_compress();
        let key = crate::utils::env::state_key().map(|raw| {
            // Derive a 32-byte AES key from the raw XINFER_STATE_KEY via SHA-256.
            let digest = sha256(&raw);
            Aes256Gcm::new_from_slice(&digest).expect("32-byte AES-256 key")
        });
        Ok(Self::new(root, compress, key))
    }

    fn path_for(&self, key: &str) -> PathBuf {
        // Sanitize the key (no path traversal).
        let safe: String = key
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
            .collect();
        self.root.join(format!("{safe}.state"))
    }

    fn encode(&self, state: &InferenceState) -> Result<Vec<u8>, StateStoreError> {
        let mut bytes = rmp_serde::to_vec(state)?;
        if self.compress {
            use std::io::Write;
            let mut out = Vec::new();
            {
                let mut encoder = zstd::stream::write::Encoder::new(&mut out, 19)
                    .map_err(|e| StateStoreError::Compress(e.to_string()))?;
                encoder.write_all(&bytes).map_err(StateStoreError::Io)?;
                encoder.finish().map_err(|e| StateStoreError::Compress(e.to_string()))?;
            }
            bytes = out;
        }
        if let Some(ref aes) = self.key {
            let nonce_bytes: [u8; 12] = rand_nonce();
            let nonce = Nonce::from_slice(&nonce_bytes);
            let ciphertext = aes
                .encrypt(nonce, bytes.as_slice())
                .map_err(|e| StateStoreError::Crypto(e.to_string()))?;
            // Prepend the 12-byte nonce so load() can decrypt.
            let mut out = Vec::with_capacity(nonce_bytes.len() + ciphertext.len());
            out.extend_from_slice(&nonce_bytes);
            out.extend_from_slice(&ciphertext);
            Ok(out)
        } else {
            Ok(bytes)
        }
    }

    fn decode(&self, raw: &[u8]) -> Result<InferenceState, StateStoreError> {
        let bytes = if self.key.is_some() {
            let (nonce_bytes, ciphertext) = raw.split_at(12);
            let nonce = Nonce::from_slice(nonce_bytes);
            self.key
                .as_ref()
                .unwrap()
                .decrypt(nonce, ciphertext)
                .map_err(|e| StateStoreError::Crypto(e.to_string()))?
        } else {
            raw.to_vec()
        };
        let bytes = if self.compress {
            use std::io::Read;
            let mut out = Vec::new();
            let mut decoder = zstd::stream::read::Decoder::new(std::io::Cursor::new(bytes))
                .map_err(|e| StateStoreError::Compress(e.to_string()))?;
            decoder
                .read_to_end(&mut out)
                .map_err(StateStoreError::Io)?;
            out
        } else {
            bytes
        };
        Ok(rmp_serde::from_slice::<InferenceState>(&bytes)?)
    }
}

impl StateStore for FsStateStore {
    fn save(&self, key: &str, state: &InferenceState) -> Result<(), StateStoreError> {
        std::fs::create_dir_all(&self.root)?;
        let encoded = self.encode(state)?;
        std::fs::write(self.path_for(key), encoded)?;
        Ok(())
    }

    fn load(&self, key: &str) -> Result<InferenceState, StateStoreError> {
        let path = self.path_for(key);
        let raw = std::fs::read(&path).map_err(|_| StateStoreError::NotFound(key.to_string()))?;
        self.decode(&raw)
    }

    fn delete(&self, key: &str) -> Result<(), StateStoreError> {
        let path = self.path_for(key);
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        Ok(())
    }

    fn list(&self) -> Result<Vec<String>, StateStoreError> {
        let mut keys = Vec::new();
        if !self.root.exists() {
            return Ok(keys);
        }
        for entry in std::fs::read_dir(&self.root)? {
            if let Some(name) = entry?.file_name().to_str() {
                if let Some(k) = name.strip_suffix(".state") {
                    keys.push(k.to_string());
                }
            }
        }
        keys.sort();
        Ok(keys)
    }
}

/// A 12-byte AES-GCM nonce (random, unique per save).
fn rand_nonce() -> [u8; 12] {
    rand::random::<[u8; 12]>()
}

/// Derive a stable 32-byte AES-256 key from the raw XINFER_STATE_KEY via SHA-256.
fn sha256(data: &[u8]) -> [u8; 32] {
    use sha2::Digest;
    sha2::Sha256::digest(data).into()
}

/// The NV dtype helper (re-export for the store's consumers).
#[allow(dead_code)]
pub fn kv_dtype_name(d: &KvCacheDtype) -> String {
    d.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_state() -> InferenceState {
        InferenceState {
            kv: vec![SerializedKvLayer {
                k_bytes: vec![1, 2, 3, 4],
                v_bytes: vec![5, 6, 7, 8],
                k_shape: vec![2, 4, 64, 128],
                v_shape: vec![2, 4, 64, 128],
            }],
            gdn: vec![SerializedGdnState {
                conv_bytes: vec![9, 10],
                recurrent_bytes: vec![11, 12],
                conv_shape: vec![1, 16],
                recurrent_shape: vec![1, 16],
            }],
            prefix_cache: SerializedPrefixCache {
                entries: vec![(0xABCD, vec![0, 1, 2])],
                block_size: 64,
            },
            kv_dtype: "bf16".to_string(),
        }
    }

    #[test]
    fn fs_state_store_roundtrip_plain() {
        let dir = std::env::temp_dir().join("state_store_test_plain");
        let _ = std::fs::remove_dir_all(&dir);
        let store = FsStateStore::new(&dir, false, None);
        store.save("seq-42", &sample_state()).expect("save");
        let loaded = store.load("seq-42").expect("load");
        assert_eq!(loaded.kv[0].k_bytes, vec![1, 2, 3, 4]);
        assert_eq!(loaded.gdn[0].conv_bytes, vec![9, 10]);
        assert_eq!(loaded.prefix_cache.entries[0].0, 0xABCD);
        assert_eq!(store.list().expect("list").len(), 1);
        store.delete("seq-42").expect("delete");
        assert!(store.load("seq-42").is_err(), "deleted key must not load");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fs_state_store_roundtrip_compressed_encrypted() {
        let dir = std::env::temp_dir().join("state_store_test_enc");
        let _ = std::fs::remove_dir_all(&dir);
        // a 32-byte AES-256 key (the XINFER_STATE_KEY derivation path)
        let key_bytes = sha256(b"test-state-key");
        let aes = Aes256Gcm::new_from_slice(&key_bytes).expect("32-byte key");
        let store = FsStateStore::new(&dir, true, Some(aes));
        store.save("seq-7", &sample_state()).expect("save");
        // the on-disk bytes must NOT contain the plaintext (encrypted + compressed)
        let raw = std::fs::read(dir.join("seq-7.state")).expect("read raw");
        assert!(!raw.windows(4).any(|w| w == b"k_by"), "plaintext must not appear on disk");
        let loaded = store.load("seq-7").expect("load");
        assert_eq!(loaded, sample_state(), "encrypted+compressed round-trip must be lossless");
        let _ = std::fs::remove_dir_all(&dir);
    }
}