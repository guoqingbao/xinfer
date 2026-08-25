# Speculative Decoding

This document describes the speculative-decoding implementation in `xinfer`.

It covers:
- the shared `Drafter` architecture and its mechanisms (MTP, DFlash)
- how the grammar firewall integrates with acceptance
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

## Grammar firewall

When a grammar is active, every emitted token is checked against the guidance
FSM, not just the anchor:

- the draft candidates are biased toward FSM-legal tokens (a batched VOB mask
  over the draft logits, or a per-position FSM walk when
  `XINFER_SPEC_GRANULAR_MASK=1`);
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
- `XINFER_SPEC_GRANULAR_MASK=1`: use the per-position FSM walk for draft
  masking instead of the batched single-VOB mask.
- `XINFER_SPEC_NO_FF=1`: debug; disable the grammar fast-forward prefix.