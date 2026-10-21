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
- The `QosController` is owned by the scheduler (the one per engine). The gain + the
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