# QoS Scheduling (colocated prefill + decode)

The engine runs prefill and decode on the same GPU (non-disaggregated). Two
kinds of requests collide for it:

- **Latency class** (agentic): short prompts, short outputs, many back-to-back
  turns. Tight TTFT + ITL SLO.
- **Throughput class** (batch): large prompts, long outputs. Latency-tolerant,
  wants high token throughput.

QoS makes the scheduler class-aware so a Throughput flood cannot starve a
Latency user, while Latency users still get full throughput when idle.

## Model

Every `Sequence` carries a `QosClass` (inferred at arrival from `max_tokens`:
short output => Latency). Each `schedule()` step computes a **contention
vector** and makes four coupled decisions:

```
                 contention = {
                     decode_load   : class-weighted # of running DECODES
                     latency_wait  : # Latency seqs in waiting
                     thr_wait      : # Throughput seqs in waiting
                     slot_fill     : running.len / max_num_seqs
                     mem_headroom  : free_blocks / total_blocks
                 }

  (1) chunk size     = cap / (1 + decode_load)          [class-weighted ITL bound]
  (2) admission order: Latency ahead of Throughput      [stable partition within class]
  (3) reservations    : Throughput gated by Latency slot + KV reserve [opt-in]
  (4) budget          : max_num_batched_tokens * conservativeness      [SGLang-style]
```

- `decode_load` weights a Latency decode at 2x a Throughput decode, so prefill
  chunks shrink harder to protect a latency user's ITL.
- With no decodes (`decode_load == 0`) the chunk is the full `cap` (today's
  behavior) - idle throughput is untouched.

## The 2-user conflict, resolved

```
 U1 (Throughput, 64K prompt / 4K out)      U2 (Latency, 300 tok / 150 out, agentic)
        submits big prefill                      submits short turns back-to-back
                 \______________________________________/
                          contention each step:
   step:  P(admit)            D(decode)            P(admit)            D(decode)
          |                   |                   |                   |
          | U2 prefill jumps | U2 decode protected| U2 prefill jumps | U2 decode protected
          | U1 big chunk     | (chunk shrunk by   | U1 chunk shrunk  |
          |                  |  U2's 2x weight)   |  (U2 decoding)   |
          v                   v                   v                   v
   - U2 TTFT bounded: its short prefill is admitted AHEAD of U1's queued chunks.
   - U2 ITL bounded:  while U2 decodes, U1's prefill chunk shrinks (2x weight).
   - U1 throughput:   when U2 is idle, U1 prefills at full cap (no regression).
   - (opt-in) U2 is guaranteed slots + KV reserve, so U1 can't wedge/evict it.
```

## Batching & limits

Batches never grow unbounded. Each step is capped by four independent ceilings;
when they're hit the scheduler preempts/swaps rather than grows.

| Ceiling | Mechanism | Logic |
|---|---|---|
| **Slots** | `active_sequence_limit()` = `min(max_num_parallel_reqs, mamba_cache_capacity)` | caps concurrent running sequences (prefill admission *and* decode). Hybrid GDN/Mamba models also cap by Mamba slots. |
| **Tokens/step** | `max_num_batched_tokens` (x `qos.conservativeness`) | caps total prefill tokens per step; admission stops when the budget is hit. |
| **KV memory** | `block_manager.can_allocate` / `can_append` | when blocks run out, the decode phase preempts (recompute) or swaps out (CPU) the oldest sequence, and evicts prefix-cache under pressure. |
| **QoS reserves** | `latency_slot_reserve` / `latency_kv_reserve_frac` (opt-in) | Throughput admissions are gated so they can't consume the Latency reserve. |

### Prefill vs decode steps
- A step is either a **prefill** step (admit chunks from `waiting`) or a **decode**
  step (advance running decodes by one token). The `is_last_prefill` guard forces
  alternation (P, D, P, D, ...) when decodes are already running, so a decode is
  never starved behind back-to-back prefills.
- Mid-prefill sequences are re-queued to `waiting` after each chunk
  (`filter_prefill_finished`); only fully-prefilled sequences stay in `running`
  and begin decoding.

### KV-pressure handling (decode phase)
When `can_append` fails (no free block for the next token):
1. **Preempt** - the sequence is marked for recompute (its KV is released).
2. **Swap out** - the oldest preempted sequence's KV moves to CPU memory
   (`try_swap_out`), when the CUDA swap path is enabled.
3. **Evict prefix cache** - if KV exceeds `KVCACHE_SWAP_THRESHOLD` (0.95),
   reusable prefix blocks are evicted to free space.

So under memory pressure the engine sheds load (preempt/swap/evict) instead of
growing the batch; a lone uncontended request still prefills at full `cap` but
can never exceed the slot/token/KV ceilings.

## Master switch

QoS is off by default (prior FIFO + static-chunk behavior). Enable it with the
`XINFER_QOS=1` env var (or `qos.enabled` in the config). The tuning knobs
below only take effect when QoS is on.

## Knobs (`EngineConfig.qos`)

| Knob | Default | Effect |
|---|---|---|
| `XINFER_QOS` (env) | off | master switch; `1` enables QoS (restores prior FIFO + static chunk when off) |
| `qos.enabled` (config) | false | OR'd with `XINFER_QOS`; either enables QoS |
| `qos.latency_weight` | 2.0 | how much a Latency decode shrinks prefill chunks |
| `qos.throughput_weight` | 1.0 | how much a Throughput decode shrinks prefill chunks |
| `qos.conservativeness` | 1.0 | per-step prefill token budget multiplier (<1 = admit less) |
| `qos.latency_slot_reserve` | off | reserve N running slots for the Latency class |
| `qos.latency_kv_reserve_frac` | off | reserve a fraction of KV blocks for the Latency class |
| `qos.latency_max_tokens` | 1024 | `max_tokens <=` this => Latency class |
| `max_prefill_chunk_tokens` | base | ceiling for the adaptive chunk |
| `min_prefill_chunk_tokens` | 256 | floor for the adaptive chunk |

## Diagnostics
- Per-step (debug): `[qos] contention={...} chunk_size=... budget=...` when a step
  has active contention.
- Per-sequence (info, conditional): at sequence end, a `[Seq N] Scheduling: ...`
  line is shown **only if an adaptive adjustment fired** for that sequence
  (chunk shrink, priority admission, reservation block, preemption, or swap).
  It reports the class, prefill-step count, and the peak contention observed.
  The stats are owned on the `Sequence` (no locks) and captured by the engine at
  finish.

## Not implemented (future)
- Full KV preemption (recompute/swap a running Throughput request to serve a
  waiting Latency one) - vLLM-style; the reservations above cover the common
  case without the recompute risk.
- Per-tenant SLO deadlines (urgency scoring) - the class model is the first
  step toward it.