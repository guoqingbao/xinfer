//! DeepSeek V4–specific layer modules.
//!
//! Isolated from the shared `layers::{moe, attention, ...}` paths so V4 hash-gate
//! routing, MLA, compressor, indexer, and HC do not leak into other models.

pub mod compressor;
pub mod decode_buffers;
pub mod hybrid_kv;
pub mod hybrid_pool;
pub mod hyper_connection;
pub mod indexer;
pub mod mla_attention;
pub mod moe_v4;
pub mod rope_cache;

pub use compressor::{CompressorDecodeState, CompressorWeights, LayerCompressionType};
pub use decode_buffers::LayerDecodeBuffers;
pub use hybrid_kv::{
    align_native_block, build_v4_cache_specs, v4_bytes_per_native_page, V4CacheKind,
    V4LayerCacheSpec, V4_NATIVE_BLOCK_SIZE,
};
pub use hybrid_pool::{V4HybridPagePool, V4LayerPages};
pub use hyper_connection::{
    hc_expand, hc_head, hc_post, hc_pre, hc_pre_norm, HcBlockWeights, HcHeadWeights,
    HcHiddenStates, HcPreState,
};
pub use indexer::{IndexerDecodeState, IndexerWeights};
pub use mla_attention::{MlaV4Attention, MlaV4Config};
pub use moe_v4::{FusedMoeMxfp4, V4Router};
pub use rope_cache::{LayerSparseKvCache, V4RopeTables};
