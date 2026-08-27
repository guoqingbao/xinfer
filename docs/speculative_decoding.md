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
`XINFER_SPEC_ADAPTIVE_K=1` enables a tiered controller that scales K with the
rolling acceptance rate (EMA + hysteresis deadband): K only grows while
acceptance justifies it, and shrinks when it doesn't. Off by default (fixed
K = the CLI count).

> **Current stall:** enabling adaptive K (`XINFER_SPEC_ADAPTIVE_K=1`) hangs the
> engine mid-generation (no error, must be killed). The controller is
> lock-free when disabled; the hang is under investigation (suspected
> interaction between the per-step variable-K verify block and the GDN snapshot
> / KV sizing, or a lock held across a GPU op). Use `XINFER_SPEC_ADAPTIVE_K=0`
> (fixed K) until resolved.

### CUDA graphs
- **Verify graph** - the target verify forward is captured and replayed
  (single-seq). The captured output is D2D-copied into a default-pool buffer so
  replay reads a stable address.
- **Draft graph** (`XINFER_SPEC_GRAPH`, default on) - the DFlash draft
  transformer is captured and replayed when the context window is full.
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
| `XINFER_SPEC_ADAPTIVE_K` | off | scale K with acceptance (currently stalled - see note) |
| `XINFER_SPEC_CONTEXT_WINDOW` | 4096 | DFlash context cap (0 = unbounded) |
| `XINFER_SPEC_GRAPH` | on | DFlash draft CUDA graph (0 = eager draft) |
| `XINFER_SPEC_MASK_OFFLOAD` | on (CUDA) | grammar mask in the fused CUDA sampler |
| `XINFER_SPEC_GRANULAR_MASK` | off | exact per-position FSM draft mask |

CLI: `--mtp <K>` (MTP), `--draft-model-id/--draft-model-path` +
`--num-speculative-tokens <K>` (DFlash). `--disable-cuda-graph` disables all
graph capture.