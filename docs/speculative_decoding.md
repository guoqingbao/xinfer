# Speculative Decoding

This document describes the speculative-decoding implementation in `xinfer`.

It covers:
- the shared `Drafter` architecture and its mechanisms (MTP, DFlash)
- DFlash v1 vs v2: what differs, how the version and backend are detected
- how the grammar firewall integrates with acceptance (with and without masking)
- the statistics and how to read them
- configuration

## Overview

Speculative decoding drafts several tokens ahead, then verifies them against the
target model in a single prefill-style forward. A step emits
`[anchor, accepted..., continuation]` instead of one token, trading a few extra
forwards for a higher tokens-per-second.

Two mechanisms are supported, selected per request and mutually exclusive
(DFlash takes priority when both are configured):

- **MTP** (`--mtp N`): drafts with the target model's own multi-token head.
  No separate weights. Only for models that ship MTP layers (Qwen3.5/3.8,
  DeepSeek-V3).
- **DFlash** (`--draft-model-id` / `--draft-model-path`): drafts with a
  separate, replicated small draft model (e.g. `z-lab/Qwen3.8-27B-DFlash2`)
  that reads the target's projected hidden states. The draft model is loaded
  per rank and is not tensor-parallel.

Both feed the same shared verify/accept/rollback core, so the grammar firewall
and the statistics are uniform across mechanisms.

## Architecture

A `Drafter` produces a proposal for one decode step; a shared core verifies it.

```
Drafter::anchor   mechanism-specific target forward + grammar-aware sample
Drafter::draft    mechanism-specific candidate generation
core              verify forward -> accept -> hybrid rollback -> emit -> stats
```

- `MtpDrafter` (core/mtp.rs) wraps the MTP head.
- `DflashDrafter` (core/dflash.rs) wraps the DFlash draft model.
- The core is `ModelRunner::run_spec_decode` (core/speculative.rs).

The verify forward runs with `is_mtp_verify` set, which is what tells hybrid
(GDN/Mamba) layers to snapshot their recurrent state so a partial rejection can
roll back to the accepted boundary.

## DFlash v1 vs v2

DFlash is a *block-diffusion* drafter: one forward pass predicts every position
of the draft block in parallel, conditioned on projected hidden states from the
target model (at `target_layer_ids`). v2 adds two small modules on top of the
same backbone:

| | DFlash v1 | DFlash v2 |
|---|---|---|
| Token pick per position | independent argmax of each position's logits | **path selector**: keep the top-K candidates per position, score adjacent pairs with a low-rank bilinear form `S_t(a,b) = U_t(b) + <A(a) * H(h_t), B(b)>`, then greedily walk the best-scoring path |
| Within-block dependencies | attention only (suffers "suffix decay": recall falls toward the end of the block) | + **grouped dynamic 2-tap convolutions** around each attention/MLP sublayer (block-local, stateless; fixes suffix decay) |
| `dflash_config` fields | `block_size`, `mask_token_id`, `target_layer_ids` | adds `selector_rank`, `selector_top_k`, `conv_kernel_size`, `conv_group_size` |
| `architectures` (config.json) | `DFlashDraftModel` | `DFlash2DraftModel` |
| Typical shape | 6 layers, block 16 | 5 layers, block 8, selector top-16/rank-256, conv taps 2/group 16 |

The selector matters because the per-position top-1 pick is often wrong while
the correct token is still inside the top-16 candidate list; walking a coherent
path through the candidates raises the accepted-prefix length without any extra
backbone pass. The convolutions absorb the short-range work that the later
attention layers would otherwise spend on, keeping recall flat to the end of the
block.

### Version detection (automatic)

`xinfer` does not need a `--dflash-version` flag: the checkpoint tells us.
`DFlashModelConfig::has_v2_components()` returns true when the config carries
v2 signals:

1. `architectures[0]` contains `"DFlash2"` (explicit tag, when present); or
2. `dflash_config` has `selector_rank` / `selector_top_k` / the conv pair.

Model construction is content-driven: the candidate selector and the grouped
convs are only built when their weights/fields exist, so a v1 checkpoint runs
the plain argmax path and a v2 checkpoint runs selector + convs, with no
user input. (SGLang/vLLM take the same approach: they build whatever modules
the draft config declares and are version-agnostic.)

### Backend dispatch (kernel vs portable)

The v2 modules have two implementations:

- **fused CUDA kernels** (attention-rs `topk` module): `dflash_grouped_conv`
  for the convolutions, and `topk_select` + `dflash_select_candidates[_masked]`
  for the selector walk (fully GPU-resident, no host sync);
- **portable candle** fallback: a block-aware tap loop for the convs and a
  sort + CPU greedy walk for the selector (works on CPU/Metal and non-CUDA
  builds).

`XINFER_DFLASH_BACKEND` selects the backend:

| value | behavior |
|---|---|
| `auto` (default) | kernels on CUDA builds, candle otherwise. v1 checkpoints are unaffected either way (they have no v2 modules). |
| `v2` | force the fused kernels (no-op without the `cuda` feature). |
| `v1` | force the portable candle path (debugging / A-B comparisons). |

The active backend is logged at drafter init
(`DFlash drafter initialized: ... version=dflash2, backend=Auto, kernels=true`).

## Inference mechanics, with and without grammar masking

Per decode step (single sequence; batched steps fall back to plain decode):

1. **Anchor**: target-model decode forward collecting hidden states at
   `target_layer_ids`; sample the anchor token (grammar-aware sampling);
   project the hidden states (`fc` + norm) and append them to the bounded
   per-sequence context window.
2. **Draft**: embed `[anchor, MASK x N]` with the target's embedding table,
   run the draft model (cross-attention over the context window; v2 adds the
   convs), take the last N rows through the target's lm_head:
   - **no grammar**: v1 -> argmax per position; v2 -> selector walk
     (fused kernel or portable, per backend).
   - **grammar active**: build a per-position allow matrix from the guidance
     FSM and gate the selection:
     - static gate (default): repeat the *current* VOB across all N rows
       (`draft_allow_repeated`). Approximate where the VOB changes across the
       run; cheap (one GPU expand).
     - exact gate (`XINFER_SPEC_GRANULAR_MASK=1`): walk a *clone* of the FSM
       over the draft argmax chain, recording each position's VOB
       (`draft_allow_walk`). One host-built `[N, vocab]` matrix, one H2D.
     - v2 backend: the allow matrix is consumed *inside* the fused
       `dflash_select_candidates_masked` kernel (disallowed candidates are
       skipped in the argmax scoring; no pre-masking, no host sync).
     - v1 backend: the logits are pre-masked (disallowed -> -inf) and the
       portable walk / argmax runs.
   - During two-phase (tool-use) grammars the FSM defers constraints until the
     reasoning-end token; while reasoning is open the VOB is all-ones, so the
     allow matrix resolves to "no gate" and the plain unmasked path runs.
3. **Verify**: one prefill-style target forward over
   `[anchor, drafts...]` (`is_mtp_verify`), collecting the hidden states
   again for the context-window refresh.
4. **Accept (grammar firewall)**: `verify_draft_masked` accepts the longest
   prefix that is *both* target-agreeing (argmax match) and FSM-legal
   (`validate_tokens`); the continuation is the grammar-forced token if the
   FSM has one, else the VOB-masked target argmax. Accepted + continuation are
   committed to the live FSM. This firewall is what makes the approximate
   (static) draft-time gate sound: any over-permissive draft token is cut at
   acceptance.
5. **Rollback + emit**: hybrid (GDN/Mamba) state is rolled back to the
   accepted boundary on partial rejection; emit
   `[anchor, accepted..., continuation]`; refresh the context window with the
   verify block's accepted rows; record per-sequence stats.

Without any grammar, steps 2/4 reduce to: draft (argmax or selector walk) and
accept-by-target-argmax-only. The output distribution is identical to plain
decoding (lossless speculation); only the speed changes.

## Grammar firewall

When a grammar is active, every emitted token is checked against the guidance
FSM, not just the anchor:

- the draft candidates are biased toward FSM-legal tokens (see the masking
  options above; DFlash v1 uses `mask_rows`/`masked_drafts` on the logits,
  DFlash v2 uses the allow-matrix gate in the selector);
- acceptance requires both target agreement and FSM legality;
- the continuation is the FSM-masked target choice or a grammar-forced token.

The FSM is advanced in lockstep with the sequence, so it never runs ahead of
the emitted tokens. A grammar therefore cannot be bypassed by speculation.

## Statistics

Per-sequence speculative statistics are reported in the end-of-sequence
performance block, labelled by the active mechanism:

```
[Seq 0] DFlash Speculation: steps=64 proposed=960 accepted=255 rate=26.6%
        avg_tok/step=5.98 grammar_bound=0 target_bound=62
```

- `proposed` / `accepted` / `rate`: draft throughput.
- `grammar_bound` / `target_bound`: how often the grammar vs. the target model
  was the binding constraint on acceptance.
- `avg_tok/step`: emitted tokens per speculative step.

## Configuration

| Flag | Meaning |
|---|---|
| `--mtp N` | enable MTP speculation, N drafts per step |
| `--draft-model-id ID` | load a DFlash draft model from HuggingFace |
| `--draft-model-path PATH` | load a DFlash draft model from a local path |
| `--num-speculative-tokens N` | DFlash drafts per step (default: block_size - 1) |

Environment:
- `XINFER_DFLASH_BACKEND=auto|v1|v2`: draft-model compute backend (default
  `auto`: fused CUDA kernels when available, portable candle otherwise).
- `XINFER_SPEC_GRANULAR_MASK=1`: use the exact per-position FSM walk for draft
  masking instead of the batched single-VOB mask.
- `XINFER_SPEC_NO_FF=1`: debug; disable the grammar fast-forward prefix.