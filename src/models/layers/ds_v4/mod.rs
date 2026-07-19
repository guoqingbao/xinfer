//! DeepSeek V4–specific layer modules.
//!
//! Isolated from the shared `layers::{moe, attention, ...}` paths so V4 hash-gate
//! routing, MLA, compressor, indexer, and HC do not leak into other models.

pub mod compressor;
pub mod hyper_connection;
pub mod indexer;
pub mod mla_attention;
pub mod moe_v4;
pub mod rope_cache;

pub use compressor::{CompressorDecodeState, CompressorWeights, LayerCompressionType};
pub use hyper_connection::{
    hc_expand, hc_head, hc_post, hc_pre, HcBlockWeights, HcHeadWeights, HcHiddenStates, HcPreState,
};
pub use indexer::{IndexerDecodeState, IndexerWeights};
pub use mla_attention::{MlaV4Attention, MlaV4Config};
pub use moe_v4::{FusedMoeMxfp4, FusedMoeW2, V4Router};
pub use rope_cache::{LayerSparseKvCache, V4RopeTables};
