# Serde Sprint (the base layer)

The serde sprint is the foundation of the 9-sprint chain. It has three feature-sets:
1. **Msgpack IPC** - replaces the runner/multi-node/transfer IPC from the unmaintained `bincode` to `rmp-serde` (MessagePack).
2. **Real-vs-synthetic prefill stats** - the prompt-stats line separates the real (uncached) prefill rate from the synthetic (cache-hit) rate.
3. **Transfer graceful shutdown** - the `Transfer` comm thread is signaled + joined on drop, so the PD comm thread does not outlive the engine.

All three are **always-on** (no env var / CLI flag); they are correctness, not optional features. There is nothing to "enable".

---

## 1. Msgpack IPC (rmp-serde)

### What changed
Every IPC boundary that previously serialized with `bincode` now serializes with
`rmp_serde` (MessagePack). The wire framing is unchanged: a 4-byte little-endian
length prefix followed by the serialized payload.

### Data-movement cardinality
- Per message: a 4-byte LE length prefix + `len` bytes of msgpack payload.
- **Payload size:** proportional to the message (a `TransferMessage` enum variant);
  typically small (control messages) to large (KV/pd payloads).
- **Sanity bound:** a receive rejects any declared length > 1 GiB
  (`transfer/comm.rs:400`).

### Diagram: the framing
```
sender                                    receiver
  |  write_all(len.to_le_bytes())  ----->  |  read_exact(len_buf[4])
  |  write_all(rmp_serde::to_vec(msg)) --> |  read_exact(msg_buf[len])
  |                                        |  rmp_serde::from_slice(msg_buf) -> msg
  |  (the ack byte is the receiver's turn) <----|
```

### Sync/async mechanics
- Synchronous, blocking I/O over a `Read + Write` stream (a Unix socket for the
  runner IPC, a TCP stream for multi-node). No async; the caller blocks on
  `write_all` / `read_exact`.
- The msgpack encode/decode is CPU-bound + fast (no allocation-heavy tree
  like JSON).

### Data lifecycle
- The payload is a transient byte buffer (a `Vec<u8>` from `to_vec`, a
  `msg_buf` from `read_exact`). It is freed after the frame is consumed. No
  persistent state.

### Locking
- Lockless with respect to the payload (the stream is owned by the caller). The
  stream itself (a `UnixStream` / `TcpStream`) is not internally locked;
  concurrency use is the caller's responsibility (the runner uses one stream per
  message in order).

### Why msgpack over bincode
- `bincode` is unmaintained (the last release predates the current Rust edition
  + has known soundness concerns in its derive). `rmp-serde` is actively
  maintained, is the de-facto standard MessagePack for Rust, + is what the
  upstream `llguidance` / `pushdown-rs` / `attention-rs` deps already use, so it
  avoids a second serialization stack in the dep graph.

### Citations / references
- MessagePack spec: <https://msgpack.org> (the binary efficient serialization format).
- `rmp-serde` crate: <https://crates.io/crates/rmp-serde> (the Rust MessagePack + serde).
- The bincode deprecation: the `bincode` crate is unmaintained (the repo is
  archived; the last release is 1.3.3).

---

## 2. Real-vs-synthetic prefill stats

### What changed
The prompt-stats log line previously reported a single "prefill rate" that mixed
the real (uncached) prefill work with the synthetic (prefix-cache-hit) work,
which made the rate look slower than the real compute. The sprint separates them:

```
Prompt: 1234 tokens in 0.42s (real 256 tok @ 609 t/s, synthetic 978 tok @ cache-hit)
```

### Data-movement cardinality
- No data movement; this is a stats/reporting change only. It reads the
  `num_cached_tokens` (the prefix-cache hit count) already tracked on the
  sequence + the wall-clock prefill duration.

### Trigger
- Emitted once per request, at the end of the prefill phase (the same point the
  old single-rate line was printed).

### Sync/async
- Synchronous log emission on the scheduler thread (no I/O).

### Data lifecycle
- The `num_cached_tokens` is a per-sequence counter (incremented on a
  prefix-cache hit, reset on a fresh prefill). It lives for the sequence's
  lifetime.

### Locking
- The counter is read under the scheduler's existing sequence lock (no new
  lock is introduced).

### Citations / references
- Prefix caching: vLLM's Automatic Prefix Caching (Kwon et al., "Efficient Memory
  Management for Large Language Model Serving with PagedAttention", SOSP 2023,
  arXiv:2309.06180). Cached-prefix tokens are not re-computed, so they
  should not count against the real prefill throughput.

---

## 3. Transfer graceful shutdown

### What changed
The `Transfer` struct (the PD prefill/decode transfer engine) spawns a
background comm thread. Previously, dropping the `Transfer` left the thread
running (a detached thread + a potential use-of-freed-state). The sprint adds:
- A `stop_flag: Arc<AtomicBool>` that the thread polls.
- A `comm_handle: Option<JoinHandle>` for the thread's join handle.
- A `Drop` impl that sets the `stop_flag` + joins the `comm_handle`.

### Diagram: the lifecycle
```
Transfer::new()
   |
   +-- spawns comm thread  ---->  loop { if stop_flag { break } ... }
   |         (comm_handle held)
   |
drop(transfer)
   |
   +-- Drop::drop: stop_flag.store(true) ; comm_handle.join()
   |         (the thread observes the flag + exits; the join reaps it)
   v
  (no leaked thread, no use-after-free)
```

### Data-movement cardinality
- No KV data movement; this is a lifecycle/shutdown fix. The comm thread moves
  control messages (the transfer requests/responses) over the existing channel.

### Trigger
- `Drop` of the `Transfer` value (the end of its scope, or an explicit drop).

### Sync/async mechanics
- The comm thread is a std `thread` (blocking). The `stop_flag` is an
  `AtomicBool` (lockless signal). The `JoinHandle::join()` blocks the dropping
  thread until the comm thread exits (a synchronous handoff).

### Data lifecycle
- The `stop_flag` + `comm_handle` live as long as the `Transfer`. On drop, the
  flag is set (one-way, monotonic, no reset), the thread is joined (reaped), +
  both are freed with the `Transfer`.

### Locking
- Lockless signal (the `AtomicBool`; the Relaxed ordering is sufficient for a
  stop signal). The `JoinHandle` uses the OS thread-join (no user-space lock).

### Citations / references
- The RAII resource cleanup: the Rust `Drop` trait (the "Drop" chapter, the
  Rust Reference, <https://doc.rust-lang.org/reference/traits.html#the-drop-trait>).
- The thread join pattern: `std::thread::JoinHandle` (the
  <https://doc.rust-lang.org/std/thread/struct.JoinHandle.html>).

---

## Usage / enablement summary

| Feature | How to enable | Default |
|---|---|---|
| Msgpack IPC | (always on; no flag) | on |
| Real-vs-synthetic prefill stats | (always on; no flag) | on |
| Transfer graceful shutdown | (always on; the `Drop` impl) | on |

There are no user-facing knobs in this sprint. It is the architectural base:
the msgpack IPC is the transport every higher layer (the scheduler, the
speculative decode, the state store, the lattice) builds on, the real-vs-
synthetic prefill stats are the signal every QoS controller (a higher layer)
consumes, + the Transfer graceful shutdown is the lifecycle contract the
disaggregated P/D (a higher layer) relies on. On this branch, those higher
layers do not yet exist; the serde sprint delivers the foundation they will
use.