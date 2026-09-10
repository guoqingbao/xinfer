# The token-Flow Contract (the inference loop relative to the token stream)

This document is the reference for the invariant that keeps the grammar FSM, the
PDA mirror, and the Sequence in lockstep through the serialized decode cycle.
It is the contract every token-production path must uphold, and the reason the
gate, the reads, and the settle are shaped the way they are. Read it before
touching `runner.rs`, `guided_decoding.rs`, `guidance.rs`, or the scheduler.

## The invariant

A Sequence may only ever hold a token that the FSM has accepted, and the FSM may
only ever be advanced by the tokens the Sequence will hold, in the order they
were acquired. There is exactly one writer (the gate) and exactly one append
site (the scheduler'spostprocess`), and they run in the same serialized engine
loop, so the FSM and the Sequence cannot drift apart.

## The serialized decode cycle

```
engine loop (one serialized tick, engine.rs)
│
├─ 1. SCHEDULE   prepare_step → scheduler.schedule() → owned_seqs (clones)
│
├─ 2. SAMPLE     run_forward / run_forward_mtp / run_forward_dflash / run_forward_spec_ff
│                └─ ModelRunner::run → ModelRunner::sample
│                     ├─ mask pull:  GuidanceState::compute_mask_or_eos → matcher.compute_mask_immut
│                     │              (NON-ADVANCING: reads the settled state, never force_bytes)
│                     ├─ sample():   LogitsProcessor::sample_processed_logits / _perseq / _perseq_masked
│                     └─ ff-read:    GuidanceState::compute_ff_tokens → matcher.compute_ff_tokens_immut
│                                    (NON-ADVANCING: reads the settled state)
│
├─ 3. COMMIT     gate_forward → ModelRunner::gate_commit(seq_ids, runs, ff)
│                └─ GuidanceState::commit_run → matcher.try_consume_tokens + matcher.settle
│                   (THE SINGLE WRITER: advances the Earley parser + settles the forced bytes)
│
└─ 4. APPEND     finish_step → scheduler.postprocess → seq.append_token
                 (THE SINGLE APPEND SITE: the Sequence is only mutable here)
```

Steps 2 and 3 are the only places the FSM is read or written. Step 2 reads are
non-advancing (they observe the state settled by the previous step's commit).
Step 3 is the single writer. Step 4 is the single append site. Because the
engine loop is serialized, the FSM state at the start of step 2 of tick N+1 is
exactly the state produced by step 3 of tick N — so the mask and the PDA are
always consistent with the Sequence.

## The gate (the single ordered-queue writer)

`ModelRunner::gate_commit(seq_ids, runs, ff)` is the one funnel every
token-production path passes through just before the append:

- `ff = false` (plain / MTP / DFlash): the whole run is committed in order via
  `commit_run` (the several-clicks `try_consume_tokens` + the `settle`), and the
  FSM-accepted prefix is returned.
- `ff = true` (spec-FF): the base run is committed, the grammar-forced
  continuation is read from the now-settled state (`ff_tokens`, non-advancing),
  and the continuation is committed — so the ordered queue `[base, ff…]` is
  appended as one serialized stream. The spec-FF path has no separate mid-step
  writer; it is the gate with `ff = true`.

The gate returns only the FSM-accepted prefix, so a rejected suffix can never
reach the append site. Unguided sequences (no FSM) pass through whole.

## Non-advancing reads (the `_immut` contract)

`compute_mask_immut` and `compute_ff_tokens_immut` derive their result from the
already-settled state (the `currently_forced_bytes`) without running
`force_bytes` and without touching the `ff_tokens_cache`. This is what makes a
read a pure observation: calling it any number of times, or interleaving it with
other reads, never moves the parser position. The position moves only in the
gate (the `commit_run` → `try_consume_tokens` + `settle`).

`settle` (the `force_bytes`) is the single position-advancing step on the read
side, and it runs once per committed batch (the end of `commit_run`), not per
token — the mask is never pulled from the state in the middle of a token stream.

## The PDA mirror (the epsilon-closure consistency)

The PDA is a fast CPU mirror of the Earley parser. It is consistent with the
parser only when both use the epsilon-closure semantics:

- advance: `PdaMachine::advance_eps` (follows the epsilon moves to the terminal
  state, then the terminal move — the no stuck call dots).
- mask: `PdaMachine::mask_at_cfg` (the epsilon-closure union of the allowed
  inputs — exactly the set for which `advance_eps` succeeds).

The naive `lookup` (no epsilon closure) gets stuck at the call dots and
disagrees with the advance, so it must not be used for the mirror. The
`proof_mask_batch_consistent_with_advance_eps` test (pushdown-rs) pins this:
over the reachable config space, `mask_at_cfg` reports exactly the inputs
`advance_eps` can consume.

The Earley parser remains the authority; the PDA is a pre-check. On disagreement
the PDA is resynced by replaying the committed tokens (`resync_pda_from_cpu`).

## Per-sequence sampling (the QoS-gated path)

When QoS is on and a decode step batches multiple sequences, the row samples
with its own `temperature` / `top_k` / `top_p` (the per-batch-row tensors) via
the attention-rs per-sequence kernels (`sampling_perseq_f32`,
`sampling_perseq_masked_f32`). Without QoS the single shared strategy is used,
unchanged. This prevents one greedy request from dragging the rest of the batch
into greedy sampling.

## The single append site

`Scheduler::postprocess` is the only place a decode token enters a Sequence.
It is reached only via `LLMEngine::finish_step`, only from the engine loop, with
the gate's returned runs. The PD-server prefill-transfer append is a separate path
(outside the decode contract).

## What would break the contract

- A second FSM writer (a `commit_run` / `try_consume_tokens` outside the gate).
- An advancing read (a `compute_mask` / `compute_ff_tokens` that runs
  `force_bytes` or touches the `ff_tokens_cache`).
- A per-token settle (the settle must run once per committed batch).
- The naive PDA `lookup` / scan in place of `advance_eps` / `mask_at_cfg`.
- A Sequence append outside `postprocess`.

Each of these re-introduces a window where the FSM, the PDA, and the Sequence can
diverge, which is the class of bug this contract exists to prevent.