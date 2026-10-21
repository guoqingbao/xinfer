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

pub const QOS_ENV: &str = "XINFER_QOS";
static QOS: OnceLock<bool> = OnceLock::new();

/// Master switch for class-aware QoS scheduling (priority admission, adaptive
/// prefill chunking, KV-headroom budgeting). Default off = prior operation.
pub fn qos_enabled() -> bool {
    *QOS.get_or_init(|| {
        env::var(QOS_ENV)
            .map(|v| matches!(v.trim(), "1" | "true" | "yes" | "TRUE" | "YES"))
            .unwrap_or(false)
    })
}

pub const STREAM_RESERVOIR_ENV: &str = "XINFER_STREAM_RESERVOIR";
static STREAM_RESERVOIR: OnceLock<bool> = OnceLock::new();

/// Master switch for the per-stream output reservoir (smooths bursty token
/// delivery). Default on; set XINFER_STREAM_RESERVOIR=0 to restore the legacy
/// immediate-push path.
pub fn stream_reservoir() -> bool {
    *STREAM_RESERVOIR.get_or_init(|| {
        env::var(STREAM_RESERVOIR_ENV)
            .map(|v| !matches!(v.trim().to_lowercase().as_str(), "0" | "false" | "no"))
            .unwrap_or(true)
    })
}

/// Cap on the DFlash projected-hidden context window kept per sequence (in rows).
/// `0` means unbounded full history; set e.g. `XINFER_SPEC_CONTEXT_WINDOW=512` to
/// bound memory on very long generations.
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

/// Opt-out: capture the DFlash draft transformer into a CUDA graph (replayed when
/// the context window is full). Default ON; set `XINFER_SPEC_GRAPH=0` to force the
/// eager draft.
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

/// XINFER_PDA_CURRENT=1: mask the CURRENT decode position on-GPU via the PDA
/// table (pda_compute_masks + pda_advance) instead of the CPU matcher. Default
/// off; requires a PDA table to have been uploaded.
pub const PDA_CURRENT_ENV: &str = "XINFER_PDA_CURRENT";
static PDA_CURRENT: OnceLock<bool> = OnceLock::new();

pub fn pda_current() -> bool {
    *PDA_CURRENT.get_or_init(|| {
        cfg!(feature = "cuda")
            && env::var(PDA_CURRENT_ENV)
                .ok()
                .map(|v| v.trim() == "1" || v.trim().eq_ignore_ascii_case("true"))
                .unwrap_or(false)
    })
}

/// Skip the `owned_seqs.len() == 1` gate so concurrent sequences each draft
/// (batched verify). Set `XINFER_SPEC_PARALLEL_DRAFT=1` (default off).
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

/// Speculative fast-forward decode: after the sampled base token, append the full
/// grammar-forced (ff) run instead of sampling it one token at a time. Set
/// `XINFER_SPEC_FF=1` to enable (default off).
pub const SPEC_FF_ENV: &str = "XINFER_SPEC_FF";
static SPEC_FF: OnceLock<bool> = OnceLock::new();

pub fn spec_ff() -> bool {
    *SPEC_FF.get_or_init(|| {
        env::var(SPEC_FF_ENV)
            .ok()
            .map(|v| matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false)
    })
}

/// Cap on how many concurrent sequences get a DFlash drafter slot when parallel
/// drafting is enabled. Sequences beyond the cap fall back to plain decode.
/// Default 2.
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

/// `XINFER_SPEC_REJECTION_SAMPLING=1` enables lossless rejection-sampling verify
/// for speculative decode (the Leviathan algorithm: accept a draft token with
/// probability `min(1, p_target/p_draft)`, on rejection sample the residual
/// `max(0, p_target - p_draft)`). Default OFF: the greedy verify (the argmax
/// accept, the no distribution correction) is used. When ON + the target sampling
/// is non-greedy (temperature > 0), the verify honors the target's sampled
/// distribution (the no biased output). The drafter's per-position logits (the
/// `p_draft`) are required -- the MTP head's `draft_tokens_gpu` returns them.
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

/// `XINFER_SPEC_DYNAMIC_K=1` enables the per-sequence, per-step adaptive draft
/// length (the rolling prefix-acceptance-curve knee). Default OFF: the draft
/// length is the fixed `--num-speculative-tokens` ceiling (the prior behavior,
/// the no surprise for existing deployments). When ON, each sequence's K tracks
/// its own drafter reliability (a weak drafter drops K, a strong one holds the
/// ceiling), so the verify-prefill cost is no longer paid for draft positions the
/// drafter can't fill.
pub const SPEC_DYNAMIC_K_ENV: &str = "XINFER_SPEC_DYNAMIC_K";
static SPEC_DYNAMIC_K: OnceLock<bool> = OnceLock::new();

pub fn spec_dynamic_k() -> bool {
    *SPEC_DYNAMIC_K.get_or_init(|| {
        env::var(SPEC_DYNAMIC_K_ENV)
            .ok()
            .map(|v| matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false)
    })
}

/// `XINFER_SPEC_GRANULAR_MASK` is NOT implemented (the no the per-position PDA mask
/// granularity for the speculative verify). The per-position draft-position masking
/// (the P3 `mask_verify_logits`) is always on when a grammar is active; there is no
/// separate "granular" mode. This const + the warning below make the unsupported
/// env var VISIBLE (the no the silent no-op the operator set expecting an effect).
pub const SPEC_GRANULAR_MASK_ENV: &str = "XINFER_SPEC_GRANULAR_MASK";

/// Warn at startup about the unsupported spec-decode env vars (the no the silent
/// no-op). Called once at engine startup (the no per-request). The operator who set
/// an unsupported var sees a clear message instead of a silently-ignored config.
pub fn warn_unsupported_spec_env_vars() {
    if env::var(SPEC_GRANULAR_MASK_ENV).is_ok() {
        crate::log_warn!(
            "{SPEC_GRANULAR_MASK_ENV} is set but NOT implemented (the per-position PDA mask \
             granularity is always on when a grammar is active, the no separate mode). \
             The var is ignored. Remove it from the config or track the feature request."
        );
    }
}

/// When set to "1" or "true", the grammar constrains all generation including
/// reasoning. Default is "0" (two-phase reasoning: unconstrained reasoning,
/// constrained response).
pub const LLG_FULL_ENV: &str = "XINFER_LLG_FULL";
static LLG_FULL: OnceLock<bool> = OnceLock::new();

pub fn llg_full_enabled() -> bool {
    *LLG_FULL.get_or_init(|| {
        env::var(LLG_FULL_ENV)
            .map(|v| matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false)
    })
}

/// `XINFER_MAX_REASONING_TOKENS=N` sets the reasoning token budget for the
/// `ModelDefault`/`Medium` reasoning level. The other levels scale from it:
/// `Low` = N/2, `High` = N*2, `ChainOfThought` (xhigh) = N*4. The budget caps
/// the `reasoning_block` rule via the `[max_tokens=N]` annotation (the hard
/// grammar backstop, not just soft sampling bias only). The `Custom` reasoning
/// level (the user-provided template) + the YAML user-defined reasoning
/// grammars are NOT capped (the user's grammar is authoritative). The `None`
/// when unset (the prior behavior, the level-specific hardcoded caps).
pub const MAX_REASONING_TOKENS_ENV: &str = "XINFER_MAX_REASONING_TOKENS";
static MAX_REASONING_TOKENS: OnceLock<Option<usize>> = OnceLock::new();

pub fn max_reasoning_tokens() -> Option<usize> {
    *MAX_REASONING_TOKENS.get_or_init(|| {
        env::var(MAX_REASONING_TOKENS_ENV)
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
    })
}

/// `XINFER_PDA_GRAMMAR=1` enables the GPU-resident PDA (pushdown-automaton)
/// grammar path. When off (default), the existing CPU parser walk is used.
pub const PDA_GRAMMAR_ENV: &str = "XINFER_PDA_GRAMMAR";
static PDA_GRAMMAR: OnceLock<bool> = OnceLock::new();

pub fn pda_grammar_enabled() -> bool {
    *PDA_GRAMMAR.get_or_init(|| {
        env::var(PDA_GRAMMAR_ENV)
            .ok()
            .map(|v| matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false)
    })
}

/// `XINFER_DEBUG_LLG=1` enables verbose llg (grammar) debug logging (the PDA
/// desync resyncs, the anti-loop kicks, the grammar init). Default off.
pub const DEBUG_LLG_ENV: &str = "XINFER_DEBUG_LLG";
static DEBUG_LLG: OnceLock<bool> = OnceLock::new();

pub fn debug_llg() -> bool {
    *DEBUG_LLG.get_or_init(|| {
        env::var(DEBUG_LLG_ENV)
            .map(|v| matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false)
    })
}

// the anti-loop detector (the loop-defense)
// The master switch (default OFF = zero overhead, the detector is not even
// constructed) + the probe-forward depth (the confirmation horizon for a stable
// loop). The parsing is factored into pure helpers (the testable primitives);
// the OnceLock getters are the process-wide caching wrappers.

/// `XINFER_ANTI_LOOP=1` enables the in-flight closed-loop detector + the
/// anti-loop kick. Default OFF (the overhead: the detector is not constructed,
/// the OBSERVE/APPLY points are no-ops).
pub const ANTI_LOOP_ENV: &str = "XINFER_ANTI_LOOP";

/// `XINFER_REPETITION_PROBE_DEPTH` (default 512): the probe-forward horizon that
/// confirms a candidate loop period is STABLE (a true closed loop keeps
/// repeating across the depth; a one-off phrase repeat does not).
pub const REPETITION_PROBE_DEPTH_ENV: &str = "XINFER_REPETITION_PROBE_DEPTH";
pub const DEFAULT_REPETITION_PROBE_DEPTH: usize = 512;

static ANTI_LOOP: OnceLock<bool> = OnceLock::new();
static REPETITION_PROBE_DEPTH: OnceLock<usize> = OnceLock::new();

/// Pure parsing primitive (the testable core): the anti-loop flag from the raw
/// env string. Inclusive: "1"/"true"/"yes" (case-insensitive) are on. Exclusive:
/// anything else (including unset) is off.
fn parse_anti_loop_flag(raw: Option<String>) -> bool {
    raw.map(|v| matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false)
}

/// Pure parsing primitive: the probe depth from the raw env string. Inclusive: a
/// positive integer is used. Exclusive: unset / non-numeric / zero all fall back
/// to the default (512) (a zero depth would confirm nothing).
fn parse_probe_depth(raw: Option<String>) -> usize {
    raw.and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_REPETITION_PROBE_DEPTH)
}

pub fn anti_loop_enabled() -> bool {
    *ANTI_LOOP.get_or_init(|| parse_anti_loop_flag(env::var(ANTI_LOOP_ENV).ok()))
}

pub fn repetition_probe_depth() -> usize {
    *REPETITION_PROBE_DEPTH
        .get_or_init(|| parse_probe_depth(env::var(REPETITION_PROBE_DEPTH_ENV).ok()))
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

/// Environment variable to disable soft masking for gradient smoothing.
/// When NOT set: soft masking is ENABLED (default behavior).
/// When set to "1", "true", or "yes": soft masking is DISABLED (hard -inf masking).
/// When set to "0", "false", or "no": soft masking is ENABLED.
pub const SOFT_MASK_DISABLED_ENV: &str = "XINFER_SOFT_MASK_DISABLED";

static SOFT_MASK_DISABLED: OnceLock<bool> = OnceLock::new();

pub fn soft_mask_disabled() -> bool {
    *SOFT_MASK_DISABLED.get_or_init(|| {
        env::var(SOFT_MASK_DISABLED_ENV)
            .map(|v| !matches!(v.trim().to_lowercase().as_str(), "0" | "false" | "no"))
            .unwrap_or(false)
    })
}
/// `XINFER_DEBUG_ACTOR=1` enables verbose logging at every actor message boundary
/// in the SchedD engine loop (schedule, forward, gate, postprocess, delivery, cleanup).
/// Default off (zero overhead).
pub const ACTOR_DEBUG_ENV: &str = "XINFER_DEBUG_ACTOR";
static ACTOR_DEBUG: OnceLock<bool> = OnceLock::new();

pub fn actor_debug() -> bool {
    *ACTOR_DEBUG.get_or_init(|| {
        env::var(ACTOR_DEBUG_ENV)
            .map(|v| matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false)
    })
}