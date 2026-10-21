# LLG Sprint (the grammar-FSM state management)

The llg sprint makes grammar-constrained decoding correct + fast, by replacing the unbounded
Earley item-set with a bounded PDA (pushdown-automaton) mirror, a single-writer gate, +
non-advancing reads. It builds on the spec-decode sprint (the dynamic draft length) + the
qos sprint (the adaptive plan).

## Feature contract (the invariants every llg change preserves)
1. **The single FSM writer** - the gate (`commit_run`) is the only place the grammar FSM
   advances; the sequence append is the only place a token enters a Sequence. No mid-step
   commits, no `deep_clone` shadow walks (which desync the FSM from the sequence position).
2. **The non-advancing reads** - the mask/ff reads (`compute_mask_immut` /
    `compute_ff_tokens`) observe the settled state; they never move the parser.
3. **The PDA consistency** - the PDA state is owned by the `Matcher` (the llguidance dpda
    feature); the `GuidanceState` queries it via `pda_config` / `project_drafts` / the
    `with_pda_machine` (no parallel CPU PDA mirror that could desync). The Earley
    parser is the authority; the `resync` rolls back to the last committed boundary on
    disagreement.

## Feature-sets

1. **The PDA (the `Matcher`-owned, no parallel CPU mirror)** (`guidance.rs`) - the PDA
   (the bounded pushdown automaton) is compiled + advanced inside the `Matcher` (the llguidance
   dpda feature). The `GuidanceState` no longer keeps a parallel `pda_ctrl`/`pda_stack` mirror
   (which could desync from the Earley authority); it queries the Matcher via `pda_config` +
   drives the batched draft projection via `project_drafts` / the `with_pda_machine`. The
   `pda_mask_words` (the `mask_at_cfg` over the frontier) + the `pda_validate` (the
   `project_drafts`) are the epsilon-closure operations (no stuck call dots). The `resync`
   rolls back to the last committed boundary when the PDA + the Earley disagree.
2. **The single-writer gate** (`commit_run`) - commits a produced token run to the FSM
   (the `try_consume_tokens` + the `settle`), returning only the FSM-passing prefix. The
   `settle` runs once per committed batch (no per-token settle).
3. **The non-advancing reads** - the `compute_mask_immut` / the `compute_ff_tokens`
   observe the settled state without moving the parser.
4. **The anti-loop detector** (`loop_detect.rs`) - detects a periodic repetition (a closed
   loop, not mere word frequency) via a probe-forward confirmation horizon; the mitigation
   is a sampling-constraint kick (the forbid token -> -inf, the seed penalty), not a
   grammar-mask perturbation. Gated on `XINFER_ANTI_LOOP` (default off, zero overhead).
5. **The full-envelope grammar** (`guidance_grammar.rs`) - the reasoning levels, the
   tool-calling, the EOS, the user-defined grammar file (the YAML). The `llg_full` mode
   (the `XINFER_LLG_FULL`) always builds the full-coverage grammar (no tools required).
6. **The PDA GPU path** (`guided_decoding.rs`) - the `pda_project_masks` (the
    `fused_project`, the K+1 VOB masks for the K draft positions) + the `pda_current_step`
    (the `fused_sample`, the current-position mask + sample + advance). Gated on
    `XINFER_PDA_GRAMMAR` (default off, the CPU parser is the authority). The GPU settled
    projection is exact only for a deterministic machine with a bounded epsilon-closure
    (the `pda_is_exact` gate: `is_deterministic && max_closure_width <= 64`); when the
    gate fails (a non-deterministic frontier or a wide closure), the projection bails and
    the caller falls back to the CPU exact frontier mask (the `pda_mask`, the union over
    the full config-set) — never an inexact GPU approximation.
7. **The grammar cache** - caches the *instantiated FSM* (the expensive part), not the
   cheapark string (the cheap part). The string composition is trivial; the FSM instantiation
   (the pre-computed projections) is what is cached.

## Enablement (default state)

| Knob | Default | Effect |
|---|---|---|
| `XINFER_LLG_FULL` | off | always build the full-coverage grammar (no tools required). |
| `XINFER_PDA_GRAMMAR` | off | the GPU PDA current-position masking (the `fused_sample`). |
| `XINFER_ANTI_LOOP` | off | the in-flight closed-loop detector + the sampling kick. |
| `XINFER_DEBUG_LLG` | off | the verbose llg debug logging (the PDA desync resyncs, the anti-loop kicks). |
| `--reasoning-grammars <file>` | - | the user-defined reasoning-grammar YAML (the levels map to inner Lark expressions). |

The grammar FSM itself is always active when a grammar is requested (the tool
outputs, the tool calling); the PDA (the Matcher-owned fast path) + the anti-loop + the full-envelope are the
opt-in accelerations/hardening.

---

## The mechanics

### The PDA (the Matcher-owned fast path, no parallel CPU mirror)
```
  Earley parser (the authority)            PDA (the fast path, inside the Matcher)
  +--------------------------+             +----------------------------+
  | the unbounded item-set     |  resync    | the bounded (ctrl, stack)  |
  | (the O(items) per token)   | <-------  | (the O(1) per token)        |
  +--------------------------+             +----------------------------+
        | the compute_mask_immut               | the pda_mask_words (the mask_at_cfg)
        v                                      v
   the authoritative mask                    the fast mask (the epsilon-closure)
```
The PDA is compiled + advanced INSIDE the `Matcher` (the llguidance dpda feature). The
`GuidanceState` does NOT keep a parallel CPU PDA mirror (a prior design did; it was removed
because a second copy could desync from the Earley authority). Instead the `GuidanceState`
queries the Matcher: `pda_mask_words` reads the Matcher's current PDA config frontier + the
`mask_at_cfg` (the epsilon-closure mask); `pda_validate` delegates to the Matcher's
`project_drafts` (the batched draft projection); `pda_is_accepting` uses the Matcher's
epsilon-closure acceptance check. If the PDA + the Earley ever disagree, the `resync` rolls
back to the last committed boundary (no separate PDA-resync function).

### The PDA token→terminal mapping (the input-alphabet bridge)
The PDA's input alphabet is the terminal indices (the `0..num_inputs`), NOT the raw token
IDs. The `token_to_terminal` map (built at the from the `terminal_token_map` inverse)
converts a raw token ID to its terminal index before the PDA `lookup`/`advance_eps`.
Without this map, the PDA fast path silently returns 0 accepted tokens for any grammar
whose terminal indices do not coincide with the raw token IDs (the common case). The
`pda_validate` (the Matcher `project_drafts`) routes draft tokens through this map.

### The PDA epsilon-closure advance (the `.pushdown-rs` `advance_eps`)
The PDA advance uses the epsilon-closure advance (the `.pushdown-rs` `advance_eps`), NOT the
single-step `lookup`. The RTN compilation inserts epsilon transitions (the call dots that
push/pop the return addresses); a single-step `lookup` stalls on those call dots (the no
direct terminal transition from the call state). The `advance_eps` closes the epsilon moves
first, so the PDA walks through the call dots correctly. This is the same primitive the
`pda_mask_words` (the `mask_at_cfg`, the epsilon-closure mask) uses, so the advance + the
mask are consistent. Both run inside the Matcher's PDA machine (the llguidance dpda feature);
the xinfer `GuidanceState` only queries the results via `pda_mask_words` / `pda_validate` /
`pda_is_accepting`.

### The reasoning-enablement decision matrix
Whether a turnreasoning_block` is prepended to the grammar is NOT a single boolean — it is a
per-request decision matrix (`resolve_reasoning_decision`), with the precedence:
```
  1. An explicit per-request reasoning_effort wins over everything:
       - Some(None)  -> reasoning OFF (the user asked for no reasoning)
       - Some(level) -> reasoning ON at that level (overrides the CLI default-off
                        + the constraint-suppression)
  2. Otherwise, a user-supplied structured constraint suppresses the default reasoning
     (no engine "think" preamble on a constrained turn) -> OFF
  3. Otherwise, the --disable-reasoning CLI (the engine default) -> OFF
  4. Otherwise -> the engine default reasoning (the llg_full-gated ModelDefault)
```
The matrix is a pure function (unit-testable row-by-row, the `reasoning_decision_matrix`
test proves all 7 rows inclusive + exclusive). The grammar cache key includes the
`reasoning_effort` so two requests that differ only in their requested level do not
collide on a cached grammar.

### The differential PDA proof (the independent-oracle)
The `pda_validate_agrees_with_matcher_authority` test feeds the same token sequences
through BOTH the PDA claimant (the `pda_validate`, the token→terminal mapped + the
epsilon-closure advance) AND the CPU authority (the `validate_tokens`, the Earley walk),
using two separate `GuidanceState` instances (no shared state, no walk does not
perturb the PDA). It asserts the accepted-prefix lengths agree on every sequence
(valid + invalid + boundary). This is the R3 independent-oracle proof: the PDA is the
claimant, the CPU Earley is the oracle, not a circular self-check.

### The single-writer gate (the seal)
```
  produced token [t1, t2, ..., tk]   (the sampled + the ff-rewritten value)
         |
         v
  commit_run(run)  -->  consume_token (the Earley)  -->  settle (the once-per-batch)
         |
         +-- returns the FSM-passing prefix [t1..tn]  (the n <= k)
         |
         v
  the sequence append (the single place a token enters a Sequence)
```
The gate is the single point where the FSM advances. No mid-step commits (which would
desync the FSM from the sequence position). The append site (the `SchedD` postprocess)
consumes ONLY the gated prefix (the `GateCommit` response), never the raw runner output —
so a token the FSM rejected (the over-allowed by an inexact mask, the sampling error) is
truncated out before it reaches the Sequence. When guidance is inactive for a sequence,
the gate fails open (the gated prefix == the raw run), preserving the unguided behavior.

The spec-FF path (`run_speculative_ff`) returns only the base token; the grammar-forced
continuation (the ff run) is appended by the same gate (the `ff=true` ordered-queue
writer), which commits the base, reads the ff from the post-base settled state, + commits
the ff — all in acquisition order. The ff is committed exactly once (in the gate), never
in `run_speculative_ff` (a prior double-commit + base-duplication defect, now removed).

### The anti-loop detector
```
  recent token window  -->  periodicity check (is the tail S S S ... for some span S?)
        |
        +-- no  -->  (no kick, zero overhead)
        |
        +-- yes -->  probe-forward (does the period hold for PROBE_DEPTH?)
                  |
                  +-- confirmed  -->  sampling kick (the forbid token -> -inf, the seed penalty)
                  +-- fluke       -->  (no kick, the false-positive guard)
```

### The data-movement cardinalities
- **The PDA state:** lives inside the `Matcher` (the llguidance dpda feature) as the bounded
  (ctrl, stack) config frontier. The xinfer side holds no separate PDA copy (not the removed
  `pda_ctrl`/`pda_stack` mirror). Tiny (no per-token allocation).
- **The GPU PDA projection:** the K+1 VOB masks for the K draft positions (the
  `fused_project`), the `batch * (K+1) * words_per_vob` U32 words.
- **The anti-loop kick:** the per-seq forbid token (a u32) + the seed span (a small Vec<u32>).

### The sync/async mechanics
- The PDA advance is synchronous (in-line with the token commit, no thread) -- it runs
  inside the Matcher's `consume_token` / `project_drafts`.
- The GPU PDA projection is a fused kernel launch (the `fused_project` / the
  `fused_sample`, the one launch does the mask + the sample + the advance).
- The anti-loop detector runs inline with the decode step (no background thread).

### The data lifecycle
- The PDA state (the Matcher's ctrl + stack frontier) lives for the sequence's
  grammar-constrained span (created at the sequence start, destroyed at the sequence end /
  the grammar completion).
- The anti-loop kick is held until the periodicity clears (the `kick_store`), then released.

### The locking
- The PDA state is per-sequence inside the Matcher (no shared lock; the sequence has its
  own config frontier).
- The anti-loop `kick_store` is a `RwLock` (read by the sampler, written by the detector).

## Known gaps (the production-grade todos)

- **The PDA state is GPU-resident across steps (the per-step H2D is avoided).**
  The `pda_seq_state` (the per-seq ctrl/stack/sp) persists as GPU tensors in
  `guided_decoding.rs`; `pda_current_step` + `pda_project_masks` gather from it
  (no H2D for existing sequences, only a one-time `Tensor::from_vec` for a
  fresh sequence). The PDA authority lives in the Matcher (the llguidance dpda feature);
  the xinfer `GuidanceState` queries it via `pda_config` / `project_drafts` / the
  `with_pda_machine` (no parallel CPU mirror to keep in sync). The `resync` rolls
  back to the last committed boundary on a PDA/Earley disagreement.
- **The GPU PDA projection is exactness-gated (the RC1 fix).** The `pda_is_exact`
  gate (`is_deterministic && max_closure_width <= 64`) decides whether the GPU settled  settled projection may be used. When it fails (a non-deterministic frontier or a
  wide closure), the projection bails + the caller uses the CPU exact frontier mask
  (the `pda_mask`, the union over the full config-set). No inexact GPU mask reaches
  the sampler or the drafter.
- **The anti-loop detector is off by default** (`XINFER_ANTI_LOOP` default off,
  zero overhead). It is not enabled in the standard deployment path, so the
  sampling-constraint kick is dormant unless the operator opts in.
- **The `/v1/grammar` mapping is now unit-tested** (the
  `grammar_type_maps_to_structured_outputs` test, the pure
  `map_grammar_type_to_structured_outputs` function). A live end-to-end test
  (the constrained generation through the shared `chat_completion` path) is the
  remaining verification.

## Citations / references
- The PDA-based grammar masking: the llguidance `dpda` module (the bounded PDA state
  machine that replaces the unbounded Earley mask), <https://github.com/guidance-ai/llguidance>.
- The epsilon-closure PDA operations: the pushdown-rs `advance_eps` / `mask_at_cfg`,
  <https://github.com/sempervictus/pushdown-rs>.
- The fused PDA kernels: the attention-rs `fused_sample` / `fused_project`,
  <https://github.com/guoqingbao/attention.rs>.
- The anti-loop / repetition detection: the repetition-aware decoding line of work
  (e.g. "Repetition-Aware Decoding for Large Language Models", the self-repetition
  suppression literature).
- The full-envelope grammar (the reasoning + the tool-calling + the EOS in one grammar):
  the guidance-ai llguidance full-envelope construction,
  <https://github.com/guidance-ai/llguidance>.