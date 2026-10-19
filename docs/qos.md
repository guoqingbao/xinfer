# QoS Sprint (the self-tuning controller + the output reservoir)

The qos sprint adds a single xInfer engine to serve mixed traffic (the latency-sensitive
agentic/short/many-turn requests alongside the throughput-oriented large-prompt/long-output
requests) without hand-tuning a static config per workload. It is a self-tuning control loop
that bounds the decode inter-token latency (ITL) under prefill load, + a per-stream output
reservoir that smooths bursty token delivery.

## Feature-sets

1. **The self-tuning QoS controller** (`core/qos.rs`, the `QosController`).
2. **The output reservoir** (`server/streaming.rs`, the `OutputReservoir`).
3. **The workload-aware prefix-cache sizing** (the scheduler's prefix-cache block budget).
4. **The python-build infra** (the ungate `EngineConfig::new`, the gate the Rust-only `api`
   module, the rename of the pyo3 ctor to `from_args`)

## Enablement (default state)

| Knob | Default | Effect |
|---|---|---|
| `XINFER_QOS` | off | the master switch for the QoS controller. When off, the scheduler runs the static degenerate plan (the unchanged prior behavior). |
| `EngineConfig.qos.enabled` | `false` | the config-file switch; `XINFER_QOS` ORs with it (either enables the controller). |
| `XINFER_STREAM_RESERVOIR` | on | the output reservoir. Set `=0` to restore the legacy immediate-push streaming. |

The controller is **off by default** (zero overhead when disabled: the `QosController` is not
even constructed, the scheduler takes the static plan path). The reservoir is **on by default**
(it is a pure delivery-smoothing layer, not a scheduling change).

---

## 1. The self-tuning QoS controller

### The control loop (the sensor -> controller -> actuator)
```
   sensor            controller (QosController)              actuator (the scheduler)
  +----------+       +------------------------------+      +----------------------+
  | step dur|-----> | integral gain on ITL error  |----> | StepPlan: chunk size, |
  | (ITL    |       |  + KV-headroom budget scale |      |  token budget, class- |
  |  proxy) |       |  + output-len class infer   |      |  ordered admission,    |
  +----------+       +------------------------------+      |  reservation gates     |
        ^                                                  +----------------------+
        |__________________________________________________________|
        (the measured ITL feeds back into the gain)
```

### How it works
- **The sensor:** each scheduler step measures the wall-clock step duration (a proxy for
  the decode ITL: a big prefill chunk slows the step + starves the running decodes) + the
  model's observed output length.
- **The controller:** an integral gain drives the prefill-chunk suppression to hold the ITL
  at the SLO (the gain rises when the ITL runs over the SLO, falls when there is headroom).
  The KV headroom scales the per-step token budget down. The measured output-length
  distribution infers the Latency class (a request whose expected output is well below the
  model's typical output is Latency-class; before any outputs are observed everything is
  Throughput, so the loop warms up safely).
- **The actuator:** the scheduler applies a pure `StepPlan` (the chunk size, the token
  budget, the class-ordered admission, the reservation gates) with no QoS branches of its
  own. When QoS is off the plan is the static degenerate one (the unchanged behavior).

### Data-movement cardinality
- No KV data movement; this is a scheduling decision. The controller reads the step
  duration (a scalar) + the output length (a scalar) + the KV headroom (a scalar), and
  emits a `StepPlan` (a few scalars: the chunk size, the budget, the admission order, the
  reservation gates).

### Trigger
- Every scheduler step (the `schedule()` call), when `XINFER_QOS` is on.

### Sync/async
- Synchronous, in-line with the scheduler step (no background thread, no I/O, no new
  lock beyond the scheduler's existing lock).

### Data lifecycle
- The `QosController` is owned by the scheduler (one per engine). The gain + the
  output-length EMA are persistent controller state (no per-request state). The
  per-seq `SchedSeqStats` (the gain start->end, the ITL vs SLO, the shrunk/prio/resv/
  preempt/swap counters) is recorded per sequence + reported at the sequence's end.

### Locking
- The controller is accessed under the scheduler's existing lock (no new lock). The
  per-seq stats are written by the scheduler thread + read by the report thread (the
  report is taken at the sequence end, after the scheduler has finished writing).

### The per-seq report (what you see)
```
[Seq 42] Scheduling: QoS[Latency] itl=48/50ms gain=0.30->0.90 prefill=10 shrunk=8 prio=3 resv=0 preempt=0 swap=0 out=128
```
- `itl=48/50ms` - the measured ITL vs the SLO (shows whether the loop is holding the target).
- `gain=0.30->0.90` - the learning trajectory (a rising gain = the ITL was under pressure
  + the loop suppressed prefill harder).
- `shrunk/prio/resv/preempt/swap` - which adjustments fired.

### Citations / references
- The integral control loop: the standard PID controller (the integral term drives the
  steady-state error to zero), <https://en.wikipedia.org/wiki/PI_controller>.
- The ITL/TTFT SLO framing: the SLO-aware LLM serving literature (the "SLO-aware
  scheduling for LLM inference", the vLLM/SGLang continuous-batching + SLO work).
- The KV-headroom budget scaling: the PagedAttention memory accounting (Kwon et al.,
  "Efficient Memory Management for Large Language Model Serving with PagedAttention",
  SOSP 2023, arXiv:2309.06180).

---

## 2. The output reservoir

### What it does
A per-stream FIFO delay line that smooths bursty token delivery into a steady drip. The
LLM produces tokens in bursts (a prefill produces nothing, then a decode step produces a
batch), which makes the SSE stream stutter. The reservoir buffers the produced chunks +
drains them at a steady cadence sized to the stream's own production rate.

### Diagram
```
  engine tokens (bursty)          reservoir (FIFO)            client (steady)
  +----------+   push    +---------------------+   drain   +----------+
  | prefill  |---------->|  [t1][t2][t3]...    |---------->| t1 t2 t3 |
  | (no tok) |           |                     |  (steady | ...steady  |
  | decode   |---------->|         [t4][t5]  |   drip)  |  drip)   |
  +----------+           +---------------------+          +----------+
```

### Enablement
- On by default. `XINFER_STREAM_RESERVOIR=0` restores the legacy immediate-push path
  (the reservoir becomes a one-chunk pass-through).

### Data-movement cardinality
- One chunk (a `String` + a `bool`) per produced token, buffered + drained. The buffer
  holds at most the burst size (the tokens produced between two drain ticks).

### Trigger
- Push: every produced token (the engine's decode). Drain: a fixed cadence timer
  (the `drain_interval`), the emit count is sized to the sustained production rate.

### Sync/async mechanics
- The push is synchronous (the producing thread appends to the FIFO). The drain is a
  timer-driven async loop (the `tokio::interval`) that pops + emits at a steady rate.
  The prime phase holds the first 0.5s of output, then builds to ~2s of buffer at 85%
  of the sustained rate, then maintains at 100%.

### Data lifecycle
- A chunk lives from the push (the token produced) to the drain (the token emitted to
  the client). The reservoir is per-stream (one per SSE response), freed when the stream
  ends.

### Locking
- The reservoir is owned by the stream task (no shared lock; the single-producer /
  single-consumer FIFO is lock-free in practice because it is confined to one task).

### Citations / references
- The token-stream smoothing: the vLLM/SGLang streaming delivery (the client-side
  buffering to smooth the bursty prefill/decode token arrival).

---

## 3. The workload-aware prefix-cache sizing

When QoS is active, the prefix cache is sized to hold every concurrent sequence at full
context (`max_num_seqs x max_model_len / block_size`, capped at the KV pool) instead of a
fixed KV fraction, because latency-oriented serving lives on reusing prefixes across
turns. When QoS is off the prior fixed-fraction sizing is used. An explicit
`--prefix-cache-max-tokens` + the PD ratios still take precedence.

### Trigger
- At engine startup (the prefix-cache block budget is computed once).

### Data-movement cardinality
- No data movement; this is a startup sizing decision (the number of prefix-cache
  blocks reserved).

### Citations / references
- The prefix caching: vLLM's Automatic Prefix Caching (the Kwon et al. PagedAttention
  paper, arXiv:2309.06180, the "prefix caching" extension).

---

## 4. The python-build infra

The earliest feature sprint introduces the python-build plumbing so every downstream
sprint inherits it:
- **Ungate `EngineConfig::new`** - the Rust constructor is available under the python
  feature (the both EngineConfig structs are field-identical, so a single constructor
  serves both).
- **Gate the Rust-only `api` module** behind `not(python)` (the python build exposes the
  engine via the pyo3 bindings in `py/mod.rs`, not the Rust `api` module).
- **Rename the pyo3 ctor to `from_args`** (to avoid a duplicate `new` now that the
  Rust `new` is ungated).

### Trigger
- Build-time only (the feature-flag compilation), no runtime effect.

### Citations / references
- The pyo3 extension-module pattern: the pyo3 docs (the
  <https://pyo3.rs>, the `#[pyclass]` + `#[new]` constructor).

## 5. Actor integration (the `SchedD` ownership)

On the actor branch, the `QosController` is owned by the `Scheduler` struct, which is
owned by the `SchedD` actor. The control loop runs inside `SchedD::handle_step`:

```
SchedD::handle_step:
  a. scheduler.schedule()
       // Inside schedule():
       //   1. Measure step duration (the ITL sensor)
       //   2. policy.record_step(step_ms)  [if QoS active]
       //   3. policy.plan(&running, &blocks) -> StepPlan
       //   4. Apply thePlan: chunk_size, token_budget, class partition,
       //      reservation gates
       //   5. KvD.allocate / KvD.may_append (block management)
  b. fwd.run(  [GPU forward pass]
  c. scheduler.postprocess(...)
       // Inside postprocess:
       //   1. Append tokens to sequences
       //   2. policy.record_output(output_len)  [when sequence finishes]
       //   3. Stamp SchedSeqStats (gain_start, gain_end, itl_ms, etc.)
  d. Delivery loop (StreamD)_tokens, deliver_token, deliver_done)
  e. scheduler.clear_finished()
  f. check_canceled()
```

The `QosConfig` is accessed via `self.scheduler.policy` (the scheduler owns it).
No additional actor hops are needed for the QoS control loop it

The per-sequence QoS report is delivered via `SchedD::get_seq_report(seq_id)` which
returns `Option<SeqReport>` (the cached tokens + the `SchedSeqStats`). The server

calls this at the end of each sequence to emit the `[Seq N] Scheduling: ...` log line.

## 6. The collection framework (the `WorkloadHistogram` + the `AdaptivePlan`)

The §1 session work adds the bounded-memory workload histogram + the adaptive plan lawlaw on top of the existing self-tuning controller. This is the sensor that the
L2-L4 layers (the unified-memory governor, the lattice router, the multi-model
restructuring) consume to adapt the runtime to the workload.

### The `WorkloadHistogram` (the per-class bounded estimators)

The histogram is a matrix of bounded-memory streaming estimators, keyed by the
(regime × class) — the bounded class set {Latency, Throughput} × {Contended,
Uncontended}. It is O(1) in the number of conversations (the no per-conversation
storage; the per-class key is the bounded set). The estimators:

| Field | Estimator | Signal |
|-------|-----------|--------|
| `itl` | the P² streaming quantile (the 5-cell reservoir, the p95) | the inter-token latency per regime×class |
| `output_len` | the 32-bucket log-spaced CDF | the output-length distribution per class |
| `batch_size` | the EMA + counter | the decode batch size per regime |
| `prefix_hit` | the EMA | the prefix-cache hit-rate per class |
| `kv_pressure` | the EMA | the KV pool pressure per regime |
| `exposed_time` | the 3×2 matrix (the work-class × the class) | the per-class exposed time (the NInfer `RequestHostTiming`) |
| `request_states` | the 8-count lifecycle | the NInfer `EngineRequestState` counts per class |
| `preemptions` | the 2-count | the per-class preemption count (the SGLang `retraction_count`) |
| `turns` | the counter | the conversation warmth (the L1 `capture_seq`) |
| `reuse_evidence` | the 256-entry ring | the NInfer real-demand signal (the `proven_frontier` + the `demand_digest` + the `adopted_count`) |

The P² estimator is the Jain & Chaudhry streaming quantile (the 5-cell reservoir,
the no full CDF storage). The `p2_p95_converges` + the `p2_p95_mixed_stream` tests
prove the p95 tracks the high-ITL stream (the inclusive: every high sample is
reflected; the exclusive: the low samples do not inflate the p95).

### The `AdaptivePlan` (the actuator law)

The `plan()` law reads the histogram + emits the actuators:

| Histogram signal | Actuator | The tunable it drives |
|------------------|----------|----------------------|
| the `itl[Contended].p95` > the SLO | the `chunk_size` ↓ | the prefill suppression (the existing `gain`) |
| the `batch_size[Contended].ema` high + the `itl[Contended].p95` low | the `parallelism_hint = UseSequenceParallelism` | the TP↔SP switch (the Shift result) |
| the `batch_size[Uncontended].ema` low + the `itl[Uncontended].p95` high | the `parallelism_hint = IncreaseTp` | the TP degree (the DistServe result) |
| the `output_len[class]` CDF median low | the `sampling.top_k` tight | the sampling width (the short-output regime) |
| the `output_len[class]` CDF median high | the `sampling.top_p` wide | the sampling width (the long-output regime) |
| the `kv_pressure[Contended].above_threshold` high | the `kv_demote_threshold` ↓ | the L2 demotion (the SGLang move-gate result) |
| the `prefix_hit[class]` high + the `turns` high | the `batch_cap` ↑ | the admission (the warm conversation, the vLLM `max_num_active_reqs` result) |
| the `preemptions[class]` high | the `batch_cap` ↓ | the admission throttle (the NInfer `retraction_count` result) |

The `AdaptivePlan` wraps the existing `StepPlan` (the 6 fields) + adds the
`batch_cap` the `sampling` override, the `parallelism_hint`, + the
`kv_demote_threshold`. The `spec_draft_len` is added by the
`spec-decode` branch (the stack). The `qos` branch lands the core histogram + the core plan.

### The two-rate sensor (the NInfer `RequestHostTiming`)

The histogram is fed by the two-rate sensor:
- The FAST-step `record_step` updates the ITL P² + the gain (the fast rate).
- The per-sequence `record_output` updates the output-length CDF + the batch-size
  EMA + the prefix-hit EMA + the turns counter (the slow rate).
- The `record_admission` (the per path) records the prefix-hit + the turns + the
  reuse-evidence digest at the admission.
- The `record_preemption` records the per-class preemption (the capacity signal).

### The NInfer `Scheduler` template (the admission + the preemption + the resume)

The three pure functions port the NInfer scheduler template:
- The `admission_allowed(bypasses, concurrency)` — the FIFO bypass allowance (the
  "at most C younger requests can overtake" rule).
- The `youngest_victim(candidates, priority_ticket)` — the preemption victim
  selection (the no materializing/finished, the `recovery_pending` + the
  `priority_ticket` guard, the youngest-by-id).
- The `should_restore(have_paused, resident_empty, restoration_event)` — the
  resume gating (the `have_paused` + the `resident_empty` OR the
  `restoration_event`).

The `admission_bypass_cap` + the `preemption_victim_youngest` + the
`resume_gating` tests prove each (the inclusive: the valid case is admitted/
selected; the exclusive: the invalid case is rejected/skipped).