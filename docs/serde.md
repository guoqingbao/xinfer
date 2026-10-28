# Serde Sprint (the base layer)

The serde sprint is the foundation of the branch chain. It has four parts:

1. **Msgpack IPC** — replaces the runner / multi-node / transfer IPC from the
   unmaintained `bincode` to `rmp-serde` (MessagePack).
2. **Compact raw-byte encoding** — the disaggregated-PD raw KV/TQ byte payloads
   are encoded as MsgPack binary (a `RawBytes` newtype), not as MsgPack integer
   arrays, so they stay compact + stay under the receive limit.
3. **Real-vs-synthetic prefill stats** — the prompt-stats line separates the real
   (uncached) prefill rate from the synthetic (prefix-cache-hit) rate.
4. **Interruptible listener join** — the `Transfer` comm thread is woken on drop
   (a socket shutdown), so a parked listener does not hang the join.

All four are always-on (no env var / CLI flag); they are correctness, not
optional features. There is nothing to "enable".

---

## 1. Msgpack IPC (rmp-serde)

### What changed
Every IPC boundary that previously serialized with `bincode` now serializes with
`rmp_serde` (MessagePack): the runner sockets (`runner/mod.rs`), the command
channel (`utils/command.rs`), the multi-node TCP fan-out (`utils/multi_node.rs`),
+ the disaggregated-PD transfer channel (`transfer/comm.rs`). The wire framing is
unchanged: a 4-byte little-endian length prefix followed by the serialized
payload.

### Data-movement cardinality
- Per message: a 4-byte LE length prefix + `len` bytes of msgpack payload.
- Payload size: proportional to the message (a `MessageType` / `TransferMessage`
  enum variant); small for control messages, large for the/pd payloads.
- Sanity bound: a receive rejects any declared length > 1 GiB
  (`transfer/comm.rs`, the `receive_message_generic` length).

### Framing
```
sender                                    receiver
  |  write_all(len.to_le_bytes())  ----->  |  read_exact(len_buf[4])
  |  write_all(rmp_serde::to_vec(msg)) --> |  read_exact(msg_buf[len])
  |                                        |  rmp_serde::from_slice(msg_buf) -> msg
```
The length prefix is self-delimiting; there is no per-message ack. The reader and
writer operate on the same stream in order (one message at a time per direction),
so the framing stays aligned without a separate.

### Sync/async mechanics
- Synchronous, blocking I/O over a `Read + Write` stream (a Unix socket for the
  runner IPC, a TCP stream for multi-node). The caller blocks on `write_all` /
  `read_exact`.
- The msgpack encode/decode is CPU-bound + fast (no allocation-heavy tree like
  JSON).

### Data lifecycle
- The payload is a transient byte buffer (a `Vec<u8>` from `to_vec`, a `msg_buf`
  from `read_exact`). It is freed after the frame is consumed. No persistent
  state.

### Locking
- Lockless with respect to the payload (the stream is owned by the caller). The
  stream itself (a `UnixStream` / `TcpStream`) is not internally locked;
  concurrent use is the caller's responsibility (the runner uses one stream per
  message in order).

### Why msgpack over bincode
- `bincode` is unmaintained (the last release predates the current Rust edition
  + has known soundness concerns in its derive). `rmp-serde` is actively
  maintained, is the de-facto standard MessagePack for Rust, + is what the
  upstream `llguidance` / `pushdown-rs` / `attention-rs` deps already use, so it
  avoids a second serialization stack in the dep graph.

### Citations / references
- MessagePack spec: <https://msgpack.org> (the binary serialization format).
- `rmp-serde` crate: <https://crates.io/crates/rmp-serde> (the Rust MessagePack
  + serde).
- The bincode deprecation: the `bincode` crate is unmaintained (the repo is
  archived; the last release is 1.3.3).

---

## 2. Compact raw-byte encoding (the `RawBytes` newtype)

### What changed
`rmp_serde` encodes a bare `Vec<u8>` as a MsgPack **array of integers** (one
element per byte), which adds roughly 50% overhead on random data. The
disaggregated-PD `RemoteTcp` transfer carries raw byte payloads (the per-layer
K/V bytes in `KVTransferHandle::RemoteTcp.layer_data`, the TurboQuant buffer
bytes in `TqLayerTcpData`), so the inflated encoding can push a message past the
1 GiB receive limit even when the original `code payload fit, and it doubles
temporary memory on both ends.

A `RawBytes(Vec<u8>)` newtype wraps those fields:
- **Serialize** (`RawBytes::serialize`): `serializer.serialize_bytes(&self.0)` —
  the MsgPack binary (the/str) form, compact + length-prefixed.
- **Deserialize** (`RawBytes::deserialize`): a visitor that accepts either a
  byte string (`visit_bytes` / `visit_byte_buf`, the new form) or an integer
  sequence (`visit_seq`, the legacy array form). The legacy-sequence tolerance
  means a message in flight from an older peer (still array form) still decodes.

The bytes-to-tensor helpers (`bytes_to_cpu_tensor`, `bytes_to_tq_cpu_tensor`)
take a `&[u8]` slice instead of a `&Vec<u8>`, so they work with both `RawBytes`
(via `as_slice`) and the legacy `Vec<u8>`.

### Why it is safe
Both PD ends run the same build, so the new binary encoding is symmetric. The
legacy-sequence deserializer is the only backward-compatibility surface, and it
exists only to tolerate an in-flight message from a pre-change peer during a
rolling upgrade.

### Citations / references
- MessagePack bin types (bin8/bin16/bin32): <https://github.com/msgpack/msgpack/blob/master/spec.md>
  (the binary extension, the compact form for opaque byte payloads).
- serde's `serialize_bytes` / `deserialize_bytes`: <https://docs.rs/serde/latest/serde/trait.Serializer.html#method.serialize_bytes>

---

## 3. Real-vs-synthetic prefill stats

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
- The counter is read under the scheduler's existing sequence lock (no new lock
  is introduced).

### Citations / references
- Prefix caching: vLLM's Automatic Prefix Caching (Kwon et al., "Efficient
  Memory Management for Large Language Model Serving with PagedAttention", SOSP
  2023, arXiv:2309.06180). Cached-prefix tokens are not re-computed, so they
  should not count against the real prefill throughput.

---

## 4. Interruptible listener join

### What changed
The `Transfer` struct (the PD prefill/decode transfer engine) spawns a
background comm thread (`run_listener_loop`). Dropping a `Transfer` sets join
that thread, but the thread spends most of its life blocked in `read_exact`
(an idle peer, no message arriving) or in `accept` / `connect`. A stop flag
checked only between messages is never be observed while the thread is
parked in a blocking socket call, so the join would hang shutdown.

The fix has three parts:

1. **A `stop_flag: Arc<AtomicBool>`** that the comm thread polls. `Transfer::drop`
   sets it with `Ordering::Release`; (mod.rs, the comm thread loads it with
   `Ordering::SeqCst` (comm.rs, the between-messages + the connection-failure
   checks).
2. **A `ShutdownHandle`** on the `Communicator`: a clone of the read socket
   (a `TcpStream` for the TCP case, a `LocalStream` for the Unix case),
   populated by `establish_connection` when the connection is made.
3. **`Communicator::close`** (comm.rs): shuts the `ShutdownHandle` down
   (`TcpStream::shutdown(Both)` for the TCP case, dropping the fd for the Unix
   case), which unblocks the parked `read` / `accept` / `connect` **without
   contending on the reader lock** (the comm thread holds that lock across a
   blocking read, so setting the reader to `None` under the lock would
   deadlock). `close` then clears the reader/writer to `None`.

`Transfer::drop` (mod.rs) runs in this order: set the `stop_flag` (Release),
call `communicator.close()` (the socket interrupt), then `comm_handle.join()`.
The join is now bounded: the socket shutdown wakes the parked blocking call, the
comm thread's next `stop_flag` check (SeqCst) sees the flag + breaks, the thread
exits, the join returns.

### Diagram: the lifecycle
```
Transfer::new()
   |
   +-- spawns comm thread  ---->  loop {
   |        (comm_handle held)      if stop_flag { break }
   |                                establish_connection()   // may block in accept/connect
   |                                loop {
   |                                    if stop_flag { break }
   |                                    receive()            // may block in read_exact
   |                                }
   |                            }
   |
drop(transfer)
   |
   +-- stop_flag.store(true, Release)
   +-- communicator.close()
   |      +-- ShutdownHandle::shutdown()   // wakes the parked read/accept/connect
   |      +-- reader/writer = None
   +-- comm_handle.join()                  // bounded bounded: the thread has exited
   v
  (no leaked thread, no hung shutdown)
```

### Data-movement cardinality
- No KV data movement; this is a lifecycle/shutdown fix. The comm thread moves
  control messages (the transfer requests/responses) over the existing channel.

### Trigger
- `Drop` of the `Transfer` value (the end of its scope, or an explicit drop).

### Sync/async mechanics
- The comm thread is a std `thread` (blocking). The `stop_flag` is an
  `AtomicBool` (lockless signal; `Release` on the writer, `SeqCst` on the
  reader). The `ShutdownHandle` performs the blocking socket call. The
  `JoinHandle::join` blocks the dropping thread until the comm thread exits (a
  synchronous handoff, now bounded by the socket interrupt).

### Data lifecycle
- The `stop_flag` + the `ShutdownHandle` + the `comm_handle` live as long as the
  `Transfer`. On drop, the flag is set (one-way, monotonic, no reset), the socket
  is shut down, the thread is joined (reaped), + all three are freed with the
  `Transfer`.

### Locking
- The `stop_flag` is lockless (the atomic). The `ShutdownHandle` is guarded by
  its own `Mutex` (the `close` takes it, the comm thread never not), so the
  interrupt does not contend with the comm thread's reader lock. The `JoinHandle`
  uses the OS thread-join (no user-space lock).

### Citations / references
- The RAII resource cleanup: the Rust `Drop` trait (the "Drop" chapter, the Rust
  Reference, <https://doc.rust-lang.org/reference/destructors.html>).
- The thread join: `std::thread::JoinHandle`
  (<https://doc.rust.org/std/thread/struct.JoinHandle.html>).
- The socket shutdown to unblock a parked read: `std::net::TcpStream::shutdown`
  (<https://doc.rust.org/std/net/struct.TcpStream.html#method.shutdown>).

---

## Usage / enablement summary

| Feature | How to enable | Default |
|---|---|---|
| Msgpack IPC | (always on; no flag) | on |
| Compact raw-byte encoding (`RawBytes`) | (always on; no flag) | on |
| Real-vs-synthetic prefill stats | (always on; no flag) | on |
| Interruptible listener join | (always on; the `Drop` impl) | on |

There are no user-facing knobs in this sprint. It is the architectural base:
the msgpack IPC is the transport every higher layer (the scheduler, the
speculative decode, the state store, the lattice) builds on, the compact
raw-byte encoding keeps the disaggregated-PD payloads within the receive limit,
the real-vs-synthetic prefill stats are the signal every QoS controller (a
higher layer) consumes, + the interruptible listener join is the lifecycle
contract the disaggregated P/D (a higher layer) relies on for clean shutdown. OnOn this branch, those higher layers do not yet exist; the serde sprint delivers
the foundation they will use.