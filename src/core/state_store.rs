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

/// The version stamp (the runtime version + the model ID + the dtype + the block size).
/// Used warm-load is rejected (the XINFER_STATESTORE_STRICT) if the stored stamp
/// doesn't match the current runtime.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct VersionStamp {
    /// The xinfer runtime version (the CARGO_PKG_VERSION).
    pub runtime_version: String,
    /// The model ID (the HuggingFace ID or the local path).
    pub model_id: String,
    /// The KV dtype (the "bf16" / the "f16" / the "f8").
    pub kv_dtype: String,
    /// The KV block size.
    pub block_size: usize,
}

/// The persistent inference state: the KV cache, the GDN/mamba recurrent state,
/// and the prefix cache. This is the unit of persistence (a snapshot of one
/// sequence's or one instance's inference state).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InferenceState {
    /// Per-layer KV, serialized as CPU byte blobs (D2H'd before save).
    pub kv: Vec<SerializedKvLayer>,
    /// GDN/mamba recurrent state (conv + ssm), empty for non-hybrid models.
    pub gdn: Vec<SerializedGdnState>,
    /// The prefix-cache hash (hash -> blocks) so a restore can re-seed it.
    pub prefix_cache: SerializedPrefixCache,
    /// The KV dtype the tensors were stored in (for H2D re-materialization).
    pub kv_dtype: String,
    /// The version stamp (the runtime version + the model ID + the dtype + the block size).
    pub version_stamp: VersionStamp,
    /// The last time this state was read or written (epoch ms). Self-tracked, not
    /// filesystem atime, so LRU eviction is backend-independent. Bumped on every save/load.
    pub last_accessed: u64,
    /// SHA-256 over the serialized state with this field zeroed. Set by `finalize`
    /// before persist; verified by `verify` on load. Proves the bytes are whole.
    pub checksum: [u8; 32],
    /// True only once `finalize` has run and the write completed. A torn/partial
    /// write (crash mid-save) leaves this false, so `load` discards it and the
    /// caller falls back to the last good checkpoint.
    pub complete: bool,
}

impl InferenceState {
    /// Compute the SHA-256 over the state with the checksum field zeroed.
    fn compute_checksum(state: &InferenceState) -> [u8; 32] {
        let mut copy = state.clone();
        copy.checksum = [0u8; 32];
        copy.last_accessed = 0; // metadata, excluded from the logical-state checksum
        copy.complete = false; // flips during finalize/verify, so exclude it
        let bytes = rmp_serde::to_vec(&copy).unwrap_or_default();
        use sha2::Digest;
        sha2::Sha256::digest(&bytes).into()
    }

    /// Finalize the state for persistence: stamp the checksum + mark complete.
    /// Call this immediately before handing the state to a `'s `save`.
    pub fn finalize(&mut self) {
        self.checksum = Self::compute_checksum(self);
        self.complete = true;
    }

    /// Verify the state is whole + complete. Returns false for a torn or partial
    /// write, in which case the caller must discard it (fall back to the last
    /// good checkpoint).
    pub fn verify(&self) -> bool {
        if !self.complete {
            return false;
        }
        Self::compute_checksum(self) == self.checksum
    }
}

/// Logical equality excludes the `last_accessed` timestamp (it changes on every
/// save/load and is not part of the state's identity).
impl PartialEq for InferenceState {
    fn eq(&self, other: &Self) -> bool {
        self.kv == other.kv
            && self.gdn == other.gdn
            && self.prefix_cache == other.prefix_cache
            && self.kv_dtype == other.kv_dtype
            && self.version_stamp == other.version_stamp
    }
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

/// A single prefix-cache entry: the hash + the block IDs + the block count (the
/// number of blocks this prefix covers, the longest-match key).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PrefixEntry {
    pub hash: u64,
    pub blocks: Vec<usize>,
    /// The number of blocks this prefix covers (the longest-match key).
    pub block_count: usize,
}

/// The prefix-cache mapping (hash -> block ids) for re-seeding on restore.
#[derive(Debug, Default, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SerializedPrefixCache {
    pub entries: Vec<PrefixEntry>,
    pub block_size: usize,
}

impl SerializedPrefixCache {
    /// Find the entry with the longest cached prefix among the given candidate
    /// hashes (the conversation's prefix hashes at several lengths). Returns the
    /// entry with the max block_count, or None if no candidate hash matches.
    pub fn longest_match(&self, candidate_hashes: &[u64]) -> Option<&PrefixEntry> {
        candidate_hashes.iter()
            .filter_map(|h| self.entries.iter().find(|e| e.hash == *h))
            .max_by_key(|e| e.block_count)
    }
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
    /// Prune the store: delete states older than `max_age_ms` (the TTL). The space
    /// watermark (`max_bytes`) is a backend-specific TODO (the full accounting).
    /// Returns the number of states deleted.
    fn prune(&self, max_age_ms: u64, _max_bytes: usize) -> Result<usize, StateStoreError> {
        let now = now_ms();
        let mut deleted = 0;
        let keys = self.list()?;
        for key in &keys {
            if let Ok(state) = self.load(key) {
                if now.saturating_sub(state.last_accessed) > max_age_ms {
                    let _ = self.delete(key);
                    deleted += 1;
                }
            }
        }
        Ok(deleted)
    }
    /// Persist a per-sequence state delta under `key` (the "seq-{id}" prefix).
    /// The delta is small (block_table + GDN bytes) and is stored as a separate
    /// file alongside the full checkpoints. Default: no-op (backends that don't
    /// support deltas can override).
    fn save_delta(&self, _key: &str, _delta: &SeqStateDelta) -> Result<(), StateStoreError> {
        Ok(())
    }
    /// Load a per-sequence state delta from `key`.
    fn load_delta(&self, _key: &str) -> Result<SeqStateDelta, StateStoreError> {
        Err(StateStoreError::NotFound(_key.to_string()))
    }
    /// A cheap clone of the store for use in spawned threads (the `Box<dyn StateStore>`
    /// is not `Clone`, so backends provide this).
    fn clone_box(&self) -> Box<dyn StateStore> {
        Box::new(NoopStateStore)
    }
}

/// A no-op store (the default for `clone_box` when the backend doesn't implement it).
struct NoopStateStore;
impl StateStore for NoopStateStore {
    fn save(&self, _: &str, _: &InferenceState) -> Result<(), StateStoreError> { Ok(()) }
    fn load(&self, key: &str) -> Result<InferenceState, StateStoreError> {
        Err(StateStoreError::NotFound(key.to_string()))
    }
    fn delete(&self, _: &str) -> Result<(), StateStoreError> { Ok(()) }
    fn list(&self) -> Result<Vec<String>, StateStoreError> { Ok(Vec::new()) }
}

/// The per-sequence state delta: the block_table + GDN slot captured one finished
/// sequence. Captured at the end of every sequence and persisted async so the
/// runtime can continue processing the next request. Only the delta from the
/// last state is synchronized (the blocks allocated/freed for this sequence,
/// the GDN recurrent for this sequence's slot).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SeqStateDelta {
    /// The sequence ID (the key for the state store).
    pub seq_id: usize,
    /// The KV block table for this sequence (the blocks that hold its KV/V).
    pub block_table: Vec<u32>,
    /// The GDN/mamba slot for this sequence (None for non-hybrid models).
    pub gdn_slot: Option<usize>,
    /// The GDN/mamba state bytes for this slot (empty for non-hybrid).
    pub gdn_bytes: Vec<u8>,
    /// The prefix-cache hash for this sequence (0 = no prefix cache entry).
    pub prefix_hash: u64,
    /// The timestamp (ms) when this delta was captured.
    pub timestamp: u64,
}

/// The state-store stats (the load/save counts + the bytes + the timings +
/// the size breakdown + the IO rates). Reported periodically at runtime.
#[derive(Debug, Clone, Default)]
pub struct StateStoreStats {
    /// The number of states loaded (the start-time).
    pub loads: usize,
    /// The number of states saved (the quit-time).
    pub saves: usize,
    /// The total bytes loaded.
    pub bytes_loaded: u64,
    /// The total bytes saved.
    pub bytes_saved: u64,
    /// The total load time (the ms).
    pub load_ms: u64,
    /// The total save time (the ms).
    pub save_ms: u64,
    /// The total size on the backend (the sum of all persisted states, the).
    pub total_size_bytes: u64,
    /// The size breakdown by type (KV bytes, GDN bytes, prefix-cache bytes).
    pub kv_size_bytes: u64,
    pub gdn_size_bytes: u64,
    pub prefix_size_bytes: u64,
    /// The number of per-sequence deltas persisted.
    pub seq_deltas: usize,
    /// The timestamp (ms) of the last stats report (for rate calculation).
    pub last_report_ms: u64,
    /// The bytes saved since the last report (for rate calculation).
    pub bytes_since_last_report: u64,
}

impl StateStoreStats {
    /// A one-line report (the admins' view).
    pub fn report(&self) -> String {
        format!(
            "state-store: {} loads ({} bytes, {}ms) + {} saves ({} bytes, {}ms) | size: {} total ({} KV, {} GDN, {} prefix) | {} seq deltaseltas",
            self.loads, self.bytes_loaded, self.load_ms,
            self.saves, self.bytes_saved, self.save_ms,
            self.total_size_bytes, self.kv_size_bytes, self.gdn_size_bytes, self.prefix_size_bytes,
            self.seq_deltas
        )
    }

    /// The save rate in MB/s since the last report (0 if no time has elapsed).
    pub fn save_rate_mbps(&self) -> f64 {
        let elapsed_ms = now_ms().saturating_sub(self.last_report_ms);
        if elapsed_ms == 0 {
            return 0.0;
        }
        self.bytes_since_last_report as f64 / (elapsed_ms as f64 / 1000.0) / 1_000_000.0
    }

    /// The load rate in MB/s since the last report (0 if no time has elapsed).
    pub fn load_rate_mbps(&self) -> f64 {
        let elapsed_ms = now_ms().saturating_sub(self.last_report_ms);
        if elapsed_ms == 0 {
            return 0.0;
        }
        // Approximate: use the total bytes_loaded / total load_ms
        if self.load_ms == 0 {
            return 0.0;
        }
        self.bytes_loaded as f64 / (self.load_ms as f64 / 1000.0) / 1_000_000.0
    }

    /// A detailed multi-line report (the periodic stats output).
    pub fn detailed_report(&self) -> String {
        let save_rate = self.save_rate_mbps();
        let load_rate = self.load_rate_mbps();
        format!(
            "[state-store] loads={} ({} bytes, {}ms, {:.2} MB/s) | saves={} ({} bytes, {}ms, {:.2} MB/s)\n\
             [state-store] size: {} total ({} KV, {} GDN, {} prefix) | {} seq deltas\n\
             [state-store] rates: save={:.2} MB/s, load={:.2} MB/s",
            self.loads, self.bytes_loaded, self.load_ms, load_rate,
            self.saves, self.bytes_saved, self.save_ms, save_rate,
            self.total_size_bytes, self.kv_size_bytes, self.gdn_size_bytes, self.prefix_size_bytes,
            self.seq_deltas,
            save_rate, load_rate
        )
    }
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
    /// The loaded state failed its integrity check (a torn or partial write). The
    /// caller must discard it and fall back to the last good checkpoint.
    #[error("corrupt state for key {0}: checksum or completeness failed")]
    Corrupt(String),
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
        let mut state = state.clone();
        state.last_accessed = now_ms();
        state.finalize(); // the checksum + the complete flag (the trait-level integrity)
        let encoded = self.encode(&state)?;
        std::fs::write(self.path_for(key), &encoded)?;
        Ok(())
    }

    fn load(&self, key: &str) -> Result<InferenceState, StateStoreError> {
        let path = self.path_for(key);
        let raw = std::fs::read(&path).map_err(|_| StateStoreError::NotFound(key.to_string()))?;
        let mut state = self.decode(&raw)?;
        state.last_accessed = now_ms();
        if !state.verify() {
            // the torn/partial write — discard it (the caller falls back to the last good checkpoint)
            return Err(StateStoreError::Corrupt(key.to_string()));
        }
        Ok(state)
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

    fn save_delta(&self, key: &str, delta: &SeqStateDelta) -> Result<(), StateStoreError> {
        std::fs::create_dir_all(&self.root)?;
        let path = self.path_for(key);
        let encoded = rmp_serde::to_vec(delta)?;
        std::fs::write(path, &encoded)?;
        Ok(())
    }

    fn load_delta(&self, key: &str) -> Result<SeqStateDelta, StateStoreError> {
        let path = self.path_for(key);
        let raw = std::fs::read(&path).map_err(|_| StateStoreError::NotFound(key.to_string()))?;
        Ok(rmp_serde::from_slice(&raw)?)
    }

    fn clone_box(&self) -> Box<dyn StateStore> {
        Box::new(FsStateStore {
            root: self.root.clone(),
            compress: self.compress,
            key: self.key.clone(),
        })
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

/// Build the optional AES-256-GCM key from the XINFER_STATE_KEY env var (the
/// SHA-256-derived 32-byte key). ` None when the var is unset/empty.
pub fn state_store_key() -> Option<Aes256Gcm> {
    crate::utils::env::state_key().and_then(|raw| {
        let digest = sha256(&raw);
        Aes256Gcm::new_from_slice(&digest).ok()
    })
}

/// The KV dtype helper (re-export for the store's consumers).
#[allow(dead_code)]
pub fn kv_dtype_name(d: &KvCacheDtype) -> String {
    d.to_string()
}

/// The current wall-clock time in milliseconds since the Unix epoch (the
/// `last_accessed` timestamp for the LRU eviction).
pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// The GPUDirect-Storage backend (the zero-copy NVMe<->GPU DMA path). Uses the
/// `cudarc::cufile` API (the `cuFile` runtime): the `cuFile` layer performs the
/// zero-copy DMA when the `nvidia-fs` kernel module + a GDS-capable NVMe are
/// present, and transparently falls back to a staged (CPU-bounce) copy otherwise.
/// Gated behind the `gds` cargo feature (the `cudarc` dependency).
#[cfg(feature = "gds")]
pub struct NvmeStateStore {
    cufile: std::sync::Arc<cudarc::cufile::Cufile>,
    /// Held to keep the CudaContext alive (the stream borrows from it); never read
    /// directly, hence the allow.
    #[allow(dead_code)]
    ctx: std::sync::Arc<cudarc::driver::CudaContext>,
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
    root: PathBuf,
    compress: bool,
    key: Option<Aes256Gcm>,
}

#[cfg(feature = "gds")]
impl NvmeStateStore {
    /// Construct the GDS store on `device`. `Cufile::new` / `CudaContext::new`
    /// fail when the `cuFile` runtime or CUDA is unavailable; callers should fall
    /// back to the `FsStateStore` in that case.
    pub fn new(
        root: impl AsRef<Path>,
        compress: bool,
        key: Option<Aes256Gcm>,
        device: usize,
    ) -> Result<Self, StateStoreError> {
        let cufile = cudarc::cufile::Cufile::new().map_err(|e| StateStoreError::Crypto(e.to_string()))?;
        let ctx = cudarc::driver::CudaContext::new(device).map_err(|e| StateStoreError::Crypto(e.to_string()))?;
        let stream = ctx.default_stream();
        Ok(Self {
            cufile,
            ctx,
            stream,
            root: root.as_ref().to_path_buf(),
            compress,
            key,
        })
    }

    /// Whether the GDS zero-copy path is actually available on this host (the
    /// `nvidia-fs` module + a GDS-capable NVMe). If false, the `cuFile` layer
    /// still works but stages through CPU memory (the no zero-copy).
    pub fn gds_active(&self) -> bool {
        // The GDS zero-copy path requires the nvidia-fs kernel module + a GDS-capable
        // NVMe. Probe the kernel module (cheap); the cuFile layer itself falls back
        // to a staged CPU copy when GDS is unavailable, so this is informational.
        std::path::Path::new("/sys/module/nvidia_fs").exists()
    }

    fn path_for(&self, key: &str) -> PathBuf {
        let safe: String = key
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
            .collect();
        self.root.join(format!("{safe}.state"))
    }

    /// Serialize + (optionally) compress + (optionally) encrypt the state into a
    /// byte buffer (the same pipeline as the FsStateStore).
    fn encode(&self, state: &InferenceState) -> Result<Vec<u8>, StateStoreError> {
        let mut bytes = rmp_serde::to_vec(state)?;
        if self.compress {
            use std::io::Write;
            let mut out = Vec::new();
            {
                let mut encoder = zstd::stream::write::Encoder::new(&mut out, 19)?;
                encoder.write_all(&bytes)?;
                encoder.finish()?;
            }
            bytes = out;
        }
        if let Some(ref aes) = self.key {
            let nonce_bytes: [u8; 12] = rand_nonce();
            let nonce = Nonce::from_slice(&nonce_bytes);
            let ciphertext = aes
                .encrypt(nonce, bytes.as_slice())
                .map_err(|e| StateStoreError::Crypto(e.to_string()))?;
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
            decoder.read_to_end(&mut out)?;
            out
        } else {
            bytes
        };
        Ok(rmp_serde::from_slice(&bytes)?)
    }
}

#[cfg(feature = "gds")]
impl StateStore for NvmeStateStore {
    fn save(&self, key: &str, state: &InferenceState) -> Result<(), StateStoreError> {
        std::fs::create_dir_all(&self.root)?;
        let mut state = state.clone();
        state.last_accessed = now_ms();
        state.finalize(); // the checksum + the complete flag (the trait-level integrity)
        let encoded = self.encode(&state)?;
        let path = self.path_for(key);
        // The cuFile path: register the file, DMA the payload into a GPU buffer,
        // then sync_write (the zero-copy GDS DMA when available, else a CPU copy).
        let file = std::fs::File::create(&path)?;
        let mut handle = self.cufile.register(file).map_err(|e| StateStoreError::Crypto(e.to_string()))?;
        let buf = self
            .stream
            .clone_htod(&encoded)
            .map_err(|e| StateStoreError::Crypto(e.to_string()))?;
        handle
            .sync_write(0, &buf)
            .map_err(|e| StateStoreError::Crypto(e.to_string()))?;
        Ok(())
    }

    fn load(&self, key: &str) -> Result<InferenceState, StateStoreError> {
        let path = self.path_for(key);
        let file = std::fs::File::open(&path).map_err(|_| StateStoreError::NotFound(key.to_string()))?;
        let file_size = file.metadata().map_err(StateStoreError::Io)?.len() as usize;
        let handle = self.cufile.register(file).map_err(|e| StateStoreError::Crypto(e.to_string()))?;
        let mut buf = self
            .stream
            .alloc_zeros::<u8>(file_size)
            .map_err(|e| StateStoreError::Crypto(e.to_string()))?;
        handle
            .sync_read(0, &mut buf)
            .map_err(|e| StateStoreError::Crypto(e.to_string()))?;
        let raw = self
            .stream
            .clone_dtoh(&buf)
            .map_err(|e| StateStoreError::Crypto(e.to_string()))?;
        let mut state = self.decode(&raw)?;
        state.last_accessed = now_ms();
        if !state.verify() {
            return Err(StateStoreError::Corrupt(key.to_string()));
        }
        Ok(state)
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

/// The S3 (object-store) backend. The state is serialized (rmp) + optionally
/// compressed (zstd) + optionally encrypted (AES-256-GCM), then uploaded to the
/// S3 bucket (the single PutObject for small payloads, the multipart upload for
/// large ones, chunked at 64MB for parallelism). Gated behind the `s3` cargo
/// feature (the rust-s3 crate, the durch/rust-s3).
#[cfg(feature = "s3")]
pub struct S3StateStore {
    bucket: Box<s3::Bucket>,
    runtime: tokio::runtime::Runtime,
    prefix: String,
    compress: bool,
    key: Option<Aes256Gcm>,
}

/// The S3 multipart chunk size (the 64MB, the parallelization unit).
#[cfg(feature = "s3")]
const S3_CHUNK: usize = 64 * 1024 * 1024;

#[cfg(feature = "s3")]
impl S3StateStore {
    pub fn new(
        region: s3::region::Region,
        credentials: s3::creds::Credentials,
        bucket_name: &str,
        prefix: &str,
        compress: bool,
        key: Option<Aes256Gcm>,
    ) -> Result<Self, StateStoreError> {
        let prefix = prefix.trim_end_matches('/').to_string();
        let bucket = s3::Bucket::new(bucket_name, region, credentials)
            .map_err(|e| StateStoreError::Crypto(e.to_string()))?;
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| StateStoreError::Crypto(e.to_string()))?;
        Ok(Self { bucket, runtime, prefix, compress, key })
    }

    fn object_key(&self, key: &str) -> String {
        let safe: String = key
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
            .collect();
        if self.prefix.is_empty() {
            format!("{safe}.state")
        } else {
            format!("{}/{}.state", self.prefix, safe)
        }
    }

    fn encode(&self, state: &InferenceState) -> Result<Vec<u8>, StateStoreError> {
        let mut bytes = rmp_serde::to_vec(state)?;
        if self.compress {
            use std::io::Write;
            let mut out = Vec::new();
            {
                let mut encoder = zstd::stream::write::Encoder::new(&mut out, 19)?;
                encoder.write_all(&bytes)?;
                encoder.finish()?;
            }
            bytes = out;
        }
        if let Some(ref aes) = self.key {
            let nonce_bytes: [u8; 12] = rand_nonce();
            let nonce = Nonce::from_slice(&nonce_bytes);
            let ciphertext = aes
                .encrypt(nonce, bytes.as_slice())
                .map_err(|e| StateStoreError::Crypto(e.to_string()))?;
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
            decoder.read_to_end(&mut out)?;
            out
        } else {
            bytes
        };
        Ok(rmp_serde::from_slice(&bytes)?)
    }
}

#[cfg(feature = "s3")]
impl StateStore for S3StateStore {
    fn save(&self, key: &str, state: &InferenceState) -> Result<(), StateStoreError> {
        let mut state = state.clone();
        state.last_accessed = now_ms();
        state.finalize(); // the checksum + the complete flag (the trait-level integrity)
        let encoded = self.encode(&state)?;
        let obj_key = self.object_key(key);
        self.runtime.block_on(async {
            self.bucket
                .put_object(&obj_key, &encoded)
                .await
                .map_err(|e| StateStoreError::Crypto(e.to_string()))?;
            Ok(())
        })
    }

    fn load(&self, key: &str) -> Result<InferenceState, StateStoreError> {
        let obj_key = self.object_key(key);
        let response = self
            .runtime
            .block_on(async { self.bucket.get_object(&obj_key).await })
            .map_err(|e| StateStoreError::Crypto(e.to_string()))?;
        let raw = response.to_vec();
        let mut state = self.decode(&raw)?;
        state.last_accessed = now_ms();
        if !state.verify() {
            return Err(StateStoreError::Corrupt(key.to_string()));
        }
        Ok(state)
    }

    fn delete(&self, key: &str) -> Result<(), StateStoreError> {
        let obj_key = self.object_key(key);
        self.runtime
            .block_on(async { self.bucket.delete_object(&obj_key).await })
            .map_err(|e| StateStoreError::Crypto(e.to_string()))?;
        Ok(())
    }

    fn list(&self) -> Result<Vec<String>, StateStoreError> {
        let mut keys = Vec::new();
        let listing = self
            .runtime
            .block_on(async { self.bucket.list(self.prefix.clone(), None).await })
            .map_err(|e| StateStoreError::Crypto(e.to_string()))?;
        for list_result in listing {
            for obj in list_result.contents {
                if let Some(name) = obj
                    .key
                    .strip_prefix(&self.prefix)
                    .and_then(|s| s.strip_suffix(".state"))
                {
                    keys.push(name.to_string());
                }
            }
        }
        keys.sort();
        Ok(keys)
    }
}

/// Construct a `StateStore` from a URL (the scheme determines the backend).
/// - `file://` or a bare path -> the `FsStateStore` (the CPU bounce buffer).
/// - `gds://` or `nvme://` -> the `NvmeStateStore` (the GDS zero-copy DMA, the `gds` feature).
/// - `s3://bucket/prefix` -> the `S3StateStore` (the object store, the `s3` feature).
pub fn state_store_from_url(
    url: &str,
    compress: bool,
    key: Option<Aes256Gcm>,
) -> Result<Box<dyn StateStore>, StateStoreError> {
    if let Some(rest) = url.strip_prefix("s3://") {
        #[cfg(feature = "s3")]
        {
            let (bucket_name, prefix) = rest.split_once('/').unwrap_or((rest, ""));
            let region = std::env::var("AWS_REGION")
                .ok()
                .and_then(|r| r.parse::<s3::region::Region>().ok())
                .unwrap_or(s3::region::Region::UsEast1);
            let credentials = s3::creds::Credentials::from_env()
                .map_err(|e| StateStoreError::Crypto(e.to_string()))?;
            return Ok(Box::new(S3StateStore::new(
                region, credentials, bucket_name, prefix, compress, key,
            )?));
        }
        #[cfg(not(feature = "s3"))]
        {
            let _ = rest;
            return Err(StateStoreError::Crypto(
                "s3:// URL requires the `s3` cargo feature".into(),
            ));
        }
    }
    if let Some(path) = url.strip_prefix("gds://").or_else(|| url.strip_prefix("nvme://")) {
        #[cfg(feature = "gds")]
        {
            // The GDS backend requires the cuFile runtime (the nvidia-fs module + a
            // GDS-capable NVMe). When it is unavailable, Cufile::new panics; catch
            // that and degrade to the CPU-bounce FS backend on the same path so a
            // missing GDS stack never takes the engine down.
            match std::panic::catch_unwind(|| NvmeStateStore::new(path, compress, key.clone(), 0)) {
                Ok(Ok(store)) => return Ok(Box::new(store)),
                Ok(Err(e)) => {
                    crate::log_warn!(
                        "[state] GDS store init failed ({e}); degrading to the FS backend at {path}"
                    );
                }
                Err(_) => {
                    crate::log_warn!(
                        "[state] GDS runtime unavailable; degrading to the FS backend at {path}"
                    );
                }
            }
        }
        #[cfg(not(feature = "gds"))]
        {
            let _ = path;
            return Err(StateStoreError::Crypto(
                "gds:// URL requires the `gds` cargo feature".into(),
            ));
        }
    }
    // the file:// or a bare path -> the FS backend
    let path = url.strip_prefix("file://").unwrap_or(url);
    Ok(Box::new(FsStateStore::new(path, compress, key)))
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
                entries: vec![PrefixEntry {
                    hash: 0xABCD,
                    blocks: vec![0, 1, 2],
                    block_count: 3,
                }],
                block_size: 64,
            },
            kv_dtype: "bf16".to_string(),
            version_stamp: VersionStamp {
                runtime_version: env!("CARGO_PKG_VERSION").to_string(),
                model_id: "test-model".to_string(),
                kv_dtype: "bf16".to_string(),
                block_size: 64,
            },
            last_accessed: now_ms(),
            checksum: [0u8; 32], // the placeholder (the save() finalizes it)
            complete: false,     // the save() sets it to true via finalize()
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
        assert_eq!(loaded.prefix_cache.entries[0].hash, 0xABCD);
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

    /// The GDS (NVMe zero-copy) round-trip. Only compiles + runs when the `gds`
    /// feature is on (the `cudarc` dependency) AND a CUDA device is available.
    /// i.e. on the GDS-enabled server, not the dev box.
    #[cfg(feature = "gds")]
    #[test]
    fn nvme_state_store_roundtrip() {
        let dir = std::env::temp_dir().join("state_store_test_nvme");
        let _ = std::fs::remove_dir_all(&dir);
        let store = NvmeStateStore::new(&dir, false, None, 0).expect("GDS store (CUDA device 0)");
        store.save("seq-1", &sample_state()).expect("save");
        let loaded = store.load("seq-1").expect("load");
        assert_eq!(loaded, sample_state(), "GDS round-trip must be lossless");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn prefix_cache_longest_match() {
        // Two entries for the same conversation at different prefix lengths: the
        // shorter (4 blocks) and the longer (16 blocks). The longest_match must
        // pick the longer one when both candidate hashes are present.
        let cache = SerializedPrefixCache {
            entries: vec![
                PrefixEntry { hash: 0xAA, blocks: vec![0, 1, 2, 3], block_count: 4 },
                PrefixEntry { hash: 0xBB, blocks: (0..16).collect(), block_count: 16 },
            ],
            block_size: 64,
        };
        // Only the short hash is a candidate -> the short entry wins.
        assert_eq!(cache.longest_match(&[0xAA]).unwrap().block_count, 4);
        // Both are candidates -> the longest (16) wins.
        assert_eq!(cache.longest_match(&[0xAA, 0xBB]).unwrap().block_count, 16);
        // No candidate matches -> None.
        assert!(cache.longest_match(&[0xCC]).is_none());
    }

    /// The finalize/verify operations: a finalized state verifies; the checksum
    /// covers the logical content (kv/gdn/prefix/dtype/stamp), not the metadata.
    #[test]
    fn finalize_verify_roundtrip() {
        let mut state = sample_state();
        assert!(!state.complete, "a fresh state is not complete");
        assert!(!state.verify(), "an incomplete state must not verify");
        state.finalize();
        assert!(state.complete, "finalize marks the state complete");
        assert!(state.verify(), "a finalized state verifies");
        // the checksum is stable across re-finalize (idempotent over the content)
        let first = state.checksum;
        state.finalize();
        assert_eq!(state.checksum, first, "finalize must be idempotent over content");
    }

    /// verify() detects a tampered logical field (the checksum no longer matches).
    #[test]
    fn verify_detects_tampered_content() {
        let mut state = sample_state();
        state.finalize();
        assert!(state.verify());
        // tamper with the logical content (the KV bytes)
        state.kv[0].k_bytes.push(0xFF);
        assert!(!state.verify(), "a tampered state must fail verification");
    }

    /// verify() detects an incomplete state (the complete flag is false).
    #[test]
    fn verify_detects_incomplete_state() {
        let mut state = sample_state();
        state.finalize();
        state.complete = false; // simulate a torn write that never completed
        assert!(!state.verify(), "an incomplete state must fail verification");
    }

    /// The double-buffer: a corrupt .current falls back to the good .last_good.
    #[test]
    fn double_buffer_fallback_to_last_good() {
        let dir = std::env::temp_dir().join("state_store_test_dbuf");
        let _ = std::fs::remove_dir_all(&dir);
        let store = FsStateStore::new(&dir, false, None);
        let good = sample_state();
        store.save("checkpoint.last_good", &good).expect("save last_good");

        // Simulate a torn .current write: save a state then corrupt its file bytes.
        store.save("checkpoint.current", &good).expect("save current");
        let cur_path = dir.join("checkpointcurrent.state"); // path_for sanitizes the key
        let mut bytes = std::fs::read(&cur_path).expect("read current");
        bytes.truncate(bytes.len() / 2); // tear the write
        std::fs::write(&cur_path, &bytes).expect("rewrite torn current");

        // load(.current) must fail verification; load(.last_good) must succeed.
        assert!(store.load("checkpoint.current").is_err(), "torn current must not load");
        let recovered = store.load("checkpoint.last_good").expect("last_good must load");
        assert_eq!(recovered, good, "fallback must recover the last good state");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The per-sequence delta save/load round-trip.
    #[test]
    fn seq_delta_roundtrip() {
        let dir = std::env::temp_dir().join("state_store_test_delta");
        let _ = std::fs::remove_dir_all(&dir);
        let store = FsStateStore::new(&dir, false, None);
        let delta = SeqStateDelta {
            seq_id: 42,
            block_table: vec![7, 12, 33],
            gdn_slot: Some(3),
            gdn_bytes: vec![1, 2, 3, 4, 5],
            prefix_hash: 0xABCD,
            timestamp: now_ms(),
        };
        store.save_delta("seq-42", &delta).expect("save_delta");
        let loaded = store.load_delta("seq-42").expect("load_delta");
        assert_eq!(loaded.seq_id, 42);
        assert_eq!(loaded.block_table, vec![7, 12, 33]);
        assert_eq!(loaded.gdn_slot, Some(3));
        assert_eq!(loaded.gdn_bytes, vec![1, 2, 3, 4, 5]);
        assert_eq!(loaded.prefix_hash, 0xABCD);
        // A non-existent key must fail.
        assert!(store.load_delta("seq-99").is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The stats report includes the new fields.
    #[test]
    fn stats_detailed_report_format() {
        let mut stats = StateStoreStats::default();
        stats.loads = 2;
        stats.saves = 5;
        stats.seq_deltas = 3;
        stats.total_size_bytes = 1024;
        stats.kv_size_bytes = 512;
        stats.gdn_size_bytes = 256;
        stats.prefix_size_bytes = 128;
        let report = stats.report();
        assert!(report.contains("5 saves"));
        assert!(report.contains("3 seq deltas"));
        assert!(report.contains("1024 total"));
        let detailed = stats.detailed_report();
        assert!(detailed.contains("[state-store]"));
        assert!(detailed.contains("rates:"));
    }
}