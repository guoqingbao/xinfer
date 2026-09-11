# QoS & Stream Smoothing: Production-Delivery Interaction

This document explains how the QoS scheduler (production-side rate control) and the
output reservoir (delivery-side smoothing) work together to ensure smooth, low-jitter
token delivery to clients despite variable internal production rates.

---

## 1. The Problem

LLM inference produces tokens at a **variable rate**:

- **Prefill phase**: fast burst (thousands of tokens/sec through the prompt)
- **Decode phase**: slower, steady (tens of tokens/sec, one at a time)
- **MTP/DFlash spec**: bursty (accept 3-4 tokens, then verify, then accept 0-1)
- **QoS adaptive**: prefill chunks shrink under decode load → production pauses

Without smoothing, the client sees: burst → stall → burst → stall. This causes
perceptually jarring text rendering (words "pop" in chunks rather than flowing).

---

## 2. Production-Side: QoS Rate Control

### 2.1 Class Assignment

Each request is assigned a QoS class at admission:

```
max_tokens ≤ 1024  →  Latency class   (agentic, short, many-turn)
max_tokens > 1024  →  Throughput class (long generation, batch)
```

### 2.2 Adaptive Chunk Sizing

The scheduler shrinks prefill chunks when decode load is high:

```
decode_load = Σ (running_decodes × class_weight)
  where class_weight = 2.0 for Latency, 1.0 for Throughput

chunk_size = cap / (1 + decode_load)   [floored at min_chunk]
```

**Effect on production rate:**
- No decodes running → full chunk (fastest prefill, bursty)
- 4 Latency decodes → load=8 → chunk shrinks to cap/9 (slower prefill, protects ITL)
- 4 Throughput decodes → load=4 → chunk shrinks to cap/5 (moderate)

This makes prefill production **rate-dependent on decode activity**, introducing
variability that the reservoir must smooth.

### 2.3 Priority Admission

When both classes have waiting requests:
- All Latency requests are admitted before any Throughput request
- FIFO within each class (no starvation)
- Prevents a Throughput flood from delaying an agentic turn

### 2.4 Conservativeness (SGLang-style)

```
effective_budget = max_num_batched_tokens × conservativeness
```

- `conservativeness = 1.0` (default): no change.
At `conservativeness = 0.7`: under load, admit fewer prefill tokens per step.
This protects running decodes from ITL spikes caused by large prefill batches.

### 2.5 Reservations (opt-in)

- `latency_slot_reserve`: N running slots that Throughput cannot cannot occupy
- `latency_kv_reserve_frac`: fraction of KV blocks reserved for Latency class

These are hard guarantees: a Throughput flood cannot evict or preempt a Latency request.

---

## 3. Delivery-Side: Output Reservoir

### 3.1 Architecture

```
Engine Loop ──push()──→ [ OutputReservoir ] ──drain()──→ SSE Stream → Client
                         (unbounded FIFO)
```

- **Fill side** (`push`): every produced token chunk is appended losslessly.
  No matter how fast the model produces, nothing is dropped.
- **Drain side** (`drain_batch`): a fixed-cadence timer (10ms) pulls chunks
  out at a controlled rate and emits them as SSE events.

### 3.2 The Proportional-Inverse-Derivative Behavior

The reservoir's emit rate is derived from the **sustained production rate**:

```
sustained_rate = total_pushed / elapsed_since_first_push
```

This is a **proportional term**: it tracks the lifetime average production rate.

The "inverse derivative" aspect: when production **stalls** (a decode step takes
longer, or QoS shrinks the pre), `total_pushed` freezes but `elapsed` keeps
growing. The sustained_rate **falls off gradually** (as 1/t), not instantly.
The emit rate slows proportionally, so the buffer drains slower slowly but never
stops abruptly. This is what smoothing effect.

Conversely, when production **bursts** (MTP accepts 4 tokens at once), the
sustained rate rises, (large_pushed jumps, elapsed barely changed), and
the emit rate ramps up smoothly over than dumping all tokens at once.

### 3.3 Three Phases

| Phase | Condition | Emit Rate | Purpose |
|-------|-----------|-----------|---------|
| **Prime** | `elapsed < 500ms` | 0 (hold) | Let the buffer fill before delivering anything |
| **Build** | `buffer < 2s × rate` | `0.85 × rate` | Emit slower slower than production → buffer grows to 2s floor |
| **Maintain** | `buffer ≥ 2s × rate` | `1.0 × rate` | Emit at production rate → buffer stays steady |

The 2-second buffer target means: even if production stalls for 2s, the

still receives a steady drip. No that, the drip slows (inverse-derivative)
rather than stopping.

### 3.4 Fractional Emit Accumulator

The drain timer fires every 10ms. The emit rate (chunks/sec) × 0.01s may be
less than 1 token. A fractional accumulator (`emit_acc`) carries the remainder:

```
emit_acc += emit_rate × 0.01
n = floor(emit_acc)
emit_acc -= n
emit n chunks this tick
```

This ensures sub-token-per-tick rates still deliver tokens at the correct long
rate without bursty 0-or-1 patterns.

---

## 4. QoS ↔ Reservoir Interaction

```
                    QoS controls                          Reservoir
                ┌─────────────────────┐            ┌──────────────────┐
  Request →     │ Adaptive chunk size │            │ Sustained rate  │
                │ Priority admission  │── tokens     │ Prime/Build/    │
                │ Conservativeness    │────────→    │ Maintain phases │
                │ Reservations        │            │ Fractional acc  │
                └─────────────────────┘            └────────┬─────────┘
                                                           │
                                                           ▼
                                                     SSE → Client
```

**How they compose:**

1. QoS makes production **variable-rate** (chunks shrink under load, MTP bursts)
2. Reservoir makes delivery **constant-rate** (smooth drip to of production pattern)
3. Without QoS: production is more uniform, reservoir does less work
4. Without reservoir: QoS-induced rate reaches the client as jitter

**The key invariant:** the reservoir never drops tokens. It only delays them.
The 2-second buffer target bounds the maximum delay latency.

---

## 5. Tuning Guide

| Constant | Default | Effect of increasing | Effect of decreasing |
|----------|---------|---------------------|---------------------|
| `RES_PRIME_MS` | 500 | Longer initial silence before first token appears | First token appears sooner (less smoothing) |
| `RES_BUFFER_TARGET_SECS` | 2.0 | Larger buffer → more stall tolerance, more added latency | Smaller buffer → less latency, more jitter visible |
| `RES_BUILD_FACTOR` | 0.85 | Slower build → buffer takes longer to reach target | Faster build → buffer reaches target sooner |
| `drain_interval` | 10ms | Coarser emit granularity | Smoother emit (more CPU) |
| `latency_weight` | 2.0 | Latency decodes shrink prefill chunks more aggressively | Less protection for Latency ITL |
| `conservativeness` | 1.0 | No effect (max) | Admits fewer prefill tokens under load (sms decode) |

**For agentic workloads** (many short turns, tight ITL SLO):
- `RES_PRIME_MS = 200` (faster first token)
- `RES_BUFFER_TARGET_SECS = 1.0` (less added latency)
- `latency_weight = 3.0` (stronger prefill chunk shrinking under Latency load)

**For long-generation workloads** (throughput-focused, jitter-tolerant):
- Defaults are fine
- `conservativeness = 0.8` to protect decode throughput from prefill bursts

---

## 6. Multi-Model Dispatch & QoS

When running multi-instance (`--d "0,1;2,3"`), the `ReplicaRouter` dispatches
each new request to the best instance:

```
score_i = active_seqs_i + queue_depth_i - adaptive_k_bonus_i + role_penalty_i
```

- **Latency class**: pick lowest score; tie-break by prefix cache hit
- **Throughput class**: pick instance with prefix hit AND lowest score
- **No prefix hit**: pure load balancing (lowest score)

The `qos_load` Mutex is updated every engine tick via `update_router_qos_load()`,
feeding live scheduler queue depths into the dispatch scoring. This ensures the
router always dispatches based on current load, not stale state.

When one instance is busy (high active_seqs + queue_depth), new requests are
routed to the idle instance. If the busy instance's load exceeds 2× the idle
instance's load, the migration trigger fires: a sequence is moved (KV + GDN
state transferred via IPC) to balance the load.

---

## 7. Observability

Per-sequence stats (shown at stream end, gated on QoS/multi-instance):

```
[Seq N] Scheduling: Latency: prefill=3 shrunk=2 prio=1 reserve_block=0 preempt=0 swap=0 | peak load/slot/mem=2.0/0.75/0.45
[Seq N] Dispatch: model @ replica-1 (KV stays, no migration)
```

- `shrunk=2`: prefill chunk was adaptive-shrunk 2 times (decode load was high)
- `prio=1`: this Latency request jumped the queue 1 time
- `peak load=2.0`: maximum class-weighted decode load observed
- `peak mem=0.45`: peak KV memory pressure (0=empty, 1=full)