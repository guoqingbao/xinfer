# Speculative Decoding (MTP + DFlash)

xInfer supports two speculative-decoding drafters, selected at startup and
**mutually exclusive** (enabling both is a startup error):

- **MTP** - a lightweight prediction head built into the target model (Qwen3.5)
  drafts future tokens from the backbone hidden states + KV cache.
- **DFlash** - a separate small draft model (v1 or v2) that reads the target
  model's projected hidden states and drafts a block of tokens in one forward.

Both draft the same verify/accept/rollback/stats core; only the draft step is
mechanism-specific. The active drafter identifies itself through one
`Drafter::name()` ("MTP", "DFlash1", "DFlash2"), which every shared surface
(graph capture/replay, per-seq stats, logging) reports.

## Pipeline (per decode step)

```
anchor   target forward -> sample anchor token
draft    MTP: head drafts K tokens | DFlash: draft model drafts K tokens
verify   one target forward over [anchor, draft_0..K-1]
accept   guided -> grammar firewall | unguided greedy -> argmax | unguided
          temperature -> rejection sampling (opt-in)
rollback GDN/Mamba state restored to the accepted boundary
emit     [anchor, accepted..., continuation]
```

K is the CLI count (fixed) or an adaptive tier (see below). The verify block
is always `[anchor, drafts]` sized to the active K.

## Mechanisms

### Verify selection
- **Guided** (grammar active): `verify_draft_masked` - a draft token is accepted
  only if BOTH the target argmax agrees AND the grammar FSM allows it; the
  continuation is the grammar-forced token when the FSM has one, else the
  FSM-masked target pick.
- **Unguided + greedy (ArgMax)**: `verify_draft_greedy` - argmax agreement.
- **Unguided + non-greedy (temperature/top-k/top-p)**: `verify_draft_rejection`
  (opt-in, `XINFER_SPEC_REJECTION_SAMPLING=1`) - accepts a draft token x with
  probability p_target(x); on rejection resamples the continuation from
  (p_target - one_hot(x))^+. Preserves the target distribution. Off by default
  (the greedy path is faster; rejection adds a per-step D2H + CPU top-k/p).

### Grammar-aware drafting
When a sequence is grammar-guided, the draft step projects the FSM allow-matrix
onto the draft logits (disallowed -> -inf) so the selector/argmax only picks
legal tokens. Static repeated-VOB projection by default; the exact per-position
FSM walk when `XINFER_SPEC_GRANULAR_MASK=1`. The anchor sampling can offload the
mask into the fused CUDA sampler (`XINFER_SPEC_MASK_OFFLOAD`, default on for CUDA
builds).

### Adaptive draft length (K)
`XINFER_SPEC_ADAPTIVE_K=1` enables a tiered controller that scales the draft
count K with the rolling acceptance rate. Off by default (fixed K = the CLI
count); when off, the decode paths never touch the controller (lock-free hot
path). Adaptive K is single-seq only; the batch verify path stays fixed-K.

**Control law.** K is a tier index over a bounded candidate set, not a free
integer. Each decode step feeds its accepted-draft length into an EMA
(alpha=0.2, initialized one below the starting tier); after a 10-step warmup,
every 5th step the controller re-evaluates:

```
tiers:  t_0 < t_1 < ... < t_n = max_k          (e.g. [1, 3, 8])
step DOWN:  ema <= t_{idx-1} - 0.5 - 0.25  ->  idx -= 1   (repeat while true)
step UP:    ema >  t_idx     - 0.5 + 0.0   ->  idx += 1   (only if no down move)
```

The down/up thresholds form a hysteresis deadband, and up and down never fire
in the same re-evaluation, so the tier cannot oscillate between adjacent
tiers. Movement is one tier per re-evaluation; a collapse walks down across
successive re-evaluations, and a recovery climbs back the same way.

**Tier set = capture set.** The candidate set comes from
`XINFER_SPEC_ADAPTIVE_TIERS` (comma list, clamped to `1..=max_k`, deduped and
sorted, `max_k` always included so the controller starts at the full tier),
defaulting to `[1, 3, max_k]` (SGLang-shaped low/mid/full). One list is the
single source of truth: `ModelRunner.adaptive_tiers` feeds both the
controller and the graph capture, so **every K the controller can select has
a pre-captured verify graph**.

**Per-tier verify graphs (the strategy).** With adaptive K on, `capture_mtp`
captures one CUDA graph per tier (verify_len = tier+1), largest-first:
- **Largest-first** because the wrapper's single capture memory pool peaks at
  the biggest graph; smaller tiers reuse it (AUTO_FREE_ON_LAUNCH). VRAM
  overhead is the biggest-tier peak, not the sum of the tiers.
- **Per-tier plans**: each tier bakes its own flashinfer prefill plan and a
  stable (default-pool) logits buffer into its graph; replay refreshes the
  plan with the current KV length before launch.
- **Exact-size replay**: a tier's verify_len must match a captured graph
  (replaying a bigger graph would write extra KV rows).
With adaptive K off, the single max-size graph is captured (today's
behavior, zero extra cost).

**Tier move = graph->graph swap.** At runtime the verify path checks
`is_mtp_captured(verify_len)`; with per-tier capture this is true for every
tier, so a tier move just replays a different pre-built graph. The previous
stall came from the flip between the fixed-size graph and a variable-size
eager forward: the two leave the flashinfer workspace / graph memory pool in
different states, and flipping mid-generation wedged the engine. The eager
verify path now only runs when graphs are genuinely unavailable
(`--disable-cuda-graph`, unsupported arch, or capture failure) - all-eager,
no flip.

**DFlash draft side.** The draft transformer graph is captured once at the
full block (`max_k+1`). At a lower tier the draft block is padded with extra
MASK rows to the captured size so the graph replays; the draft attention is
causal, so the first K rows are exact and the draft logits are narrowed back
to K before token selection. No per-tier draft graphs are needed (the draft
model is small; the target verify is the expensive side). The MTP draft head
(1 layer) runs its K steps eagerly - negligible.

### CUDA graphs
- **Verify graph** - the target verify forward is captured and replayed
  (single-seq). The captured output is D2D-copied into a default-pool buffer so
  replay reads a stable address. With adaptive K, one graph is captured per
  tier (verify_len = tier+1), largest-first, sharing the wrapper's single
  memory pool (peak = biggest tier, not the sum) - see the adaptive-K section
  above for the full strategy.
- **Draft graph** (`XINFER_SPEC_GRAPH`, default on) - the DFlash draft
  transformer is captured and replayed when the context window is full. The
  draft block is padded to the captured size so it replays at every tier.
- Disable all graph capture with `--disable-cuda-graph` (eager only).

### DFlash context window
The DFlash projected-hidden context is capped to the last N rows
(`XINFER_SPEC_CONTEXT_WINDOW`, default 4096; 0 = unbounded). The context is only seeded
near the prefill end (prefill position check), so the draft model attends within
its training window and memory stays bounded on long generations.

### YARN-scaled draft RoPE
When the backbone uses YARN scaling, the DFlash draft model's rotary embedding
is built with the same YARN math so its positional encoding stays consistent
with the target at extended context lengths.

### Per-sequence statistics
Replaces the old global per-step "DFlash Stats"/"MTP Stats" spam. Each sequence
accumulates a window (steps, proposed, accepted, rate, avg_tok/step,
grammar_bound, target_bound) reported once at sequence end:
`[Seq N] <Mechanism> Speculation: steps=.. proposed=.. accepted=.. rate=..%
avg_tok/step=.. grammar_bound=.. target_bound=..`.

## Tuning knobs

| Var | Default | Effect |
|---|---|---|
| `XINFER_SPEC_REJECTION_SAMPLING` | off | distribution-correct verify for non-greedy targets |
| `XINFER_SPEC_ADAPTIVE_K` | off | scale K with acceptance (per-tier verify graphs - no graph/eager flip) |
| `XINFER_SPEC_ADAPTIVE_TIERS` | `[1, 3, max_k]` | adaptive-K tier/capture set (comma list, max_k always included) |
| `XINFER_SPEC_CONTEXT_WINDOW` | 4096 | DFlash context cap (0 = unbounded) |
| `XINFER_SPEC_GRAPH` | on | DFlash draft CUDA graph (0 = eager draft) |
| `XINFER_SPEC_MASK_OFFLOAD` | on (CUDA) | grammar mask in the fused CUDA sampler |
| `XINFER_SPEC_GRANULAR_MASK` | off | exact per-position FSM draft mask |

CLI: `--mtp <K>` (MTP), `--draft-model-id/--draft-model-path` +
`--num-speculative-tokens <K>` (DFlash). `--disable-cuda-graph` disables all
graph capture.