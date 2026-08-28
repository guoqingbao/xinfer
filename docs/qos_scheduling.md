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
                 contentionion = {
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

## Knobs (`EngineConfig.qos`)

| Knob | Default | Effect |
|---|---|---|
| `qos.enabled` | true | master switch (false = legacy FIFO + static chunk) |
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