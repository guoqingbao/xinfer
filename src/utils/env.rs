use std::env;
use std::sync::OnceLock;

pub const MAMBA_SNAPSHOT_BLOCK_STRIDE_ENV: &str = "XINFER_MAMBA_SNAPSHOT_STRIDE_BLOCKS";

pub const STREAM_AS_REASONING_CONTENT_ENV: &str = "XINFER_STREAM_AS_REASONING_CONTENT";

pub const SM90_LOWER_PRECISION_GDN_PREFILL_ENV: &str = "SM90_LOWER_PRECISION_GDN_PREFILL";

static STREAM_AS_REASONING_CONTENT: OnceLock<bool> = OnceLock::new();
static SM90_LOWER_PRECISION_GDN_PREFILL: OnceLock<bool> = OnceLock::new();

pub fn sm90_lower_precision_gdn_prefill() -> bool {
    *SM90_LOWER_PRECISION_GDN_PREFILL.get_or_init(|| {
        env::var(SM90_LOWER_PRECISION_GDN_PREFILL_ENV)
            .map(|v| matches!(v.trim(), "1" | "true" | "yes" | "TRUE" | "YES"))
            .unwrap_or(false)
    })
}

pub fn stream_as_reasoning_content() -> bool {
    *STREAM_AS_REASONING_CONTENT.get_or_init(|| {
        env::var(STREAM_AS_REASONING_CONTENT_ENV)
            .map(|v| !matches!(v.trim().to_lowercase().as_str(), "0" | "false" | "no"))
            .unwrap_or(true)
    })
}

pub fn mamba_snapshot_block_stride_blocks(default: usize) -> usize {
    let default = default.max(1);
    let Ok(raw) = env::var(MAMBA_SNAPSHOT_BLOCK_STRIDE_ENV) else {
        return default;
    };
    match raw.trim().parse::<usize>() {
        Ok(0) => {
            crate::log_warn!(
                "{} must be >= 1, got 0. Falling back to default {}.",
                MAMBA_SNAPSHOT_BLOCK_STRIDE_ENV,
                default
            );
            default
        }
        Ok(v) => v,
        Err(_) => {
            crate::log_warn!(
                "Invalid {}='{}'. Falling back to default {}.",
                MAMBA_SNAPSHOT_BLOCK_STRIDE_ENV,
                raw,
                default
            );
            default
        }
    }
}

pub const DEFAULT_REASONING_MAX_TOKENS_ENV: &str = "XINFER_DEFAULT_REASONING_MAX_TOKENS";
pub const DEFAULT_REASONING_MAX_TOKENS_VALUE: usize = 512;

static DEFAULT_REASONING_MAX_TOKENS: OnceLock<usize> = OnceLock::new();

pub fn default_reasoning_max_tokens() -> usize {
    *DEFAULT_REASONING_MAX_TOKENS.get_or_init(|| {
        env::var(DEFAULT_REASONING_MAX_TOKENS_ENV)
            .map(|raw| {
                raw.trim()
                    .parse::<usize>()
                    .map(|n| {
                        if n == 0 {
                            DEFAULT_REASONING_MAX_TOKENS_VALUE
                        } else {
                            n
                        }
                    })
                    .unwrap_or(DEFAULT_REASONING_MAX_TOKENS_VALUE)
            })
            .unwrap_or(DEFAULT_REASONING_MAX_TOKENS_VALUE)
    })
}

/// Environment variable to enable llguidance debug logging.
/// Defaults to true when not set.
pub const DEBUG_LLG_ENV: &str = "XINFER_DEBUG_LLG";

static DEBUG_LLG: OnceLock<bool> = OnceLock::new();

pub fn debug_llg() -> bool {
    *DEBUG_LLG.get_or_init(|| {
        env::var(DEBUG_LLG_ENV)
            .map(|v| matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false)
    })
}

/// Environment variable to enable full-envelope grammar mode (BOS->EOS constraint).
/// When set to "1" or "true", the grammar constrains all generation including reasoning.
/// Default is "0" (two-phase reasoning: unconstrained reasoning, constrained response).
pub const LLG_FULL_ENV: &str = "XINFER_LLG_FULL";

static LLG_FULL: OnceLock<bool> = OnceLock::new();

pub fn llg_full_enabled() -> bool {
    *LLG_FULL.get_or_init(|| {
        env::var(LLG_FULL_ENV)
            .map(|v| matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false)
    })
}

/// Debug: use the granular (per-position FSM-walk) draft mask instead of the batched
/// single-VOB mask. For precise gating when the mask changes across the draft run.
pub const SPEC_GRANULAR_MASK_ENV: &str = "XINFER_SPEC_GRANULAR_MASK";

static SPEC_GRANULAR_MASK: OnceLock<bool> = OnceLock::new();

pub fn spec_granular_mask() -> bool {
    *SPEC_GRANULAR_MASK.get_or_init(|| {
        env::var(SPEC_GRANULAR_MASK_ENV)
            .map(|v| matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false)
    })
}

/// Whether grammar VOB masking is offloaded to the CUDA sampler (the mask is passed to
/// `sample_cuda_masked` and applied inside the fused top-k stage) instead of biasing the
/// logits on the CPU via `where_cond`. Default: offload on CUDA builds. Set
/// `XINFER_SPEC_MASK_OFFLOAD=0` to force the CPU (where_cond) path.
pub const SPEC_MASK_OFFLOAD_ENV: &str = "XINFER_SPEC_MASK_OFFLOAD";

static SPEC_MASK_OFFLOAD: OnceLock<bool> = OnceLock::new();

pub fn spec_mask_offload() -> bool {
    *SPEC_MASK_OFFLOAD.get_or_init(|| {
        cfg!(feature = "cuda")
            && !matches!(
                env::var(SPEC_MASK_OFFLOAD_ENV)
                    .ok()
                    .as_deref()
                    .map(|v| v.trim().eq_ignore_ascii_case("0") || v.trim().eq_ignore_ascii_case("false")),
                Some(true)
            )
    })
}

/// Use the VOB bitset path for grammar sampling (8x less data than F32 mask).
/// Requires `XINFER_SPEC_MASK_OFFLOAD=1` (default on CUDA) AND this flag.
pub const VOB_SAMPLING_ENV: &str = "XINFER_VOB_SAMPLING";

static VOB_SAMPLING: OnceLock<bool> = OnceLock::new();

pub fn vob_sampling() -> bool {
    *VOB_SAMPLING.get_or_init(|| {
        cfg!(feature = "cuda")
            && env::var(VOB_SAMPLING_ENV)
                .ok()
                .map(|v| v.trim() == "1" || v.trim().eq_ignore_ascii_case("true"))
                .unwrap_or(false)
    })
}

/// Cap on the DFlash projected-hidden context window kept per sequence (in rows).
/// `0` means unbounded full history, matching the original DFlash branch;
/// set e.g. `XINFER_SPEC_CONTEXT_WINDOW=512` to bound memory on very long generations.
pub const SPEC_CONTEXT_WINDOW_ENV: &str = "XINFER_SPEC_CONTEXT_WINDOW";

static SPEC_CONTEXT_WINDOW: OnceLock<usize> = OnceLock::new();

pub fn spec_context_window() -> usize {
    *SPEC_CONTEXT_WINDOW.get_or_init(|| {
        env::var(SPEC_CONTEXT_WINDOW_ENV)
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(4096) // Default: 4096 (matches DFlash2 training window)
    })
}

/// Opt-out: capture the DFlash draft transformer into a CUDA graph (replayed when the context
/// window is full). Default ON; set `XINFER_SPEC_GRAPH=0` to force the eager draft.
pub const SPEC_GRAPH_ENV: &str = "XINFER_SPEC_GRAPH";

static SPEC_GRAPH: OnceLock<bool> = OnceLock::new();

pub fn spec_graph() -> bool {
    *SPEC_GRAPH.get_or_init(|| {
        !matches!(
            env::var(SPEC_GRAPH_ENV)
                .ok()
                .as_deref()
                .map(|v| v.trim().eq_ignore_ascii_case("0") || v.trim().eq_ignore_ascii_case("false")),
            Some(true)
        )
    })
}

/// Master switch for the per-stream output reservoir, which smooths bursty token
/// delivery into a steady drip (decouples production from client delivery).
/// Default ON; set `XINFER_STREAM_RESERVOIR=0` to restore the legacy immediate-push
/// streaming behavior.
pub const STREAM_RESERVOIR_ENV: &str = "XINFER_STREAM_RESERVOIR";

static STREAM_RESERVOIR: OnceLock<bool> = OnceLock::new();

pub fn stream_reservoir() -> bool {
    *STREAM_RESERVOIR.get_or_init(|| {
        !matches!(
            env::var(STREAM_RESERVOIR_ENV)
                .ok()
                .as_deref()
                .map(|v| v.trim().eq_ignore_ascii_case("0") || v.trim().eq_ignore_ascii_case("false")),
            Some(true)
        )
    })
}

/// Opt-in: adaptive draft length (K scales with the rolling acceptance rate).
/// Default OFF (fixed K = the CLI `--num-speculative-tokens|mtp` count). Set
/// `XINFER_SPEC_ADAPTIVE_K=1` to enable the tiered controller.
pub const SPEC_ADAPTIVE_K_ENV: &str = "XINFER_SPEC_ADAPTIVE_K";

static SPEC_ADAPTIVE_K: OnceLock<bool> = OnceLock::new();

pub fn spec_adaptive_k() -> bool {
    *SPEC_ADAPTIVE_K.get_or_init(|| {
        env::var(SPEC_ADAPTIVE_K_ENV)
            .ok()
            .map(|v| matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false)
    })
}

/// Opt-in: allow speculative drafting for multi-sequence (parallel) decode steps.
/// Default OFF - drafting runs only for a single active sequence (the
/// `owned_seqs.len() == 1` gate). Set `XINFER_SPEC_PARALLEL_DRAFT=1` to skip
/// that gate so concurrent sequences each draft (batched verify).
pub const SPEC_PARALLEL_DRAFT_ENV: &str = "XINFER_SPEC_PARALLEL_DRAFT";

static SPEC_PARALLEL_DRAFT: OnceLock<bool> = OnceLock::new();

pub fn spec_parallel_draft() -> bool {
    *SPEC_PARALLEL_DRAFT.get_or_init(|| {
        env::var(SPEC_PARALLEL_DRAFT_ENV)
            .ok()
            .map(|v| matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false)
    })
}

/// Cap on how many concurrent sequences get a DFlash drafter slot when parallel
/// drafting is enabled (`XINFER_SPEC_PARALLEL_DRAFT`). Sequences beyond the cap
/// fall back to plain decode. Default 2.
pub const DFLASH_PARALLEL_SLOTS_ENV: &str = "XINFER_DFLASH_PARALLEL_SLOTS";

static DFLASH_PARALLEL_SLOTS: OnceLock<usize> = OnceLock::new();

pub fn dflash_parallel_slots() -> usize {
    *DFLASH_PARALLEL_SLOTS.get_or_init(|| {
        env::var(DFLASH_PARALLEL_SLOTS_ENV)
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(2)
    })
}

/// Opt-in: use rejection-sampling verify for non-greedy (temperature) targets, which
/// preserves the target distribution. Default OFF, in which case all unguided targets use
/// the fast greedy (argmax-agreement) verify (the 483 path). Set
/// `XINFER_SPEC_REJECTION_SAMPLING=1` to enable distribution-correct sampling.
pub const SPEC_REJECTION_SAMPLING_ENV: &str = "XINFER_SPEC_REJECTION_SAMPLING";

static SPEC_REJECTION_SAMPLING: OnceLock<bool> = OnceLock::new();

pub fn spec_rejection_sampling() -> bool {
    *SPEC_REJECTION_SAMPLING.get_or_init(|| {
        env::var(SPEC_REJECTION_SAMPLING_ENV)
            .ok()
            .map(|v| matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false)
    })
}

/// Optional tier set for adaptive K: a comma-separated list of draft counts
/// (e.g. `XINFER_SPEC_ADAPTIVE_TIERS=1,3,8`). Values are clamped to
/// `1..=max_k`, deduped and sorted, and `max_k` is always included so the
/// controller starts at the full tier. Default (unset): `[1, 3, max_k]`
/// (SGLang-shaped low/mid/full). The tier set is the capture set: one verify
/// CUDA graph is captured per tier, so a tier move is a graph->graph swap.
pub const SPEC_ADAPTIVE_TIERS_ENV: &str = "XINFER_SPEC_ADAPTIVE_TIERS";

static SPEC_ADAPTIVE_TIERS: OnceLock<Option<Vec<usize>>> = OnceLock::new();

pub fn spec_adaptive_tiers() -> Option<Vec<usize>> {
    SPEC_ADAPTIVE_TIERS.get_or_init(|| {
        env::var(SPEC_ADAPTIVE_TIERS_ENV)
            .ok()
            .map(|v| {
                v.split(',')
                    .filter_map(|t| t.trim().parse::<usize>().ok())
                    .collect::<Vec<_>>()
            })
            .filter(|v: &Vec<usize>| !v.is_empty())
    })
    .clone()
}

/// Master switch for class-aware QoS scheduling (priority admission,
/// class-weighted chunk sizing, reservations, conservativeness). Default OFF,
/// which restores the prior FIFO + static-chunk behavior. Set
/// `XINFER_QOS=1` to enable. (The config field `qos.enabled` is an OR'd
/// alternative for config-file users.)
pub const QOS_ENV: &str = "XINFER_QOS";

static QOS: OnceLock<bool> = OnceLock::new();

pub fn qos_enabled() -> bool {
    *QOS.get_or_init(|| {
        env::var(QOS_ENV)
            .ok()
            .map(|v| matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false)
    })
}

/// `XINFER_DFA_GRAMMAR=1` enables the GPU-resident DFA grammar path.
/// When off (default), the existing CPU parser walk is used.
pub const DFA_GRAMMAR_ENV: &str = "XINFER_DFA_GRAMMAR";

static DFA_GRAMMAR: OnceLock<bool> = OnceLock::new();

pub fn dfa_grammar_enabled() -> bool {
    *DFA_GRAMMAR.get_or_init(|| {
        env::var(DFA_GRAMMAR_ENV)
            .ok()
            .map(|v| matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false)
    })
}
