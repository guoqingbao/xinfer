# Actor Pattern: xinfer Engine Architecture

## Overview

The xinfer engine uses an actor-based concurrency model built on the
`ractor` crate (a pure-Rust Erlang/OTP-inspired actor framework). Each
domain is a single-threaded actor (a tokio task) that owns its state and
communicates exclusively via message passing. No shared mutable state
exists between actors.

This replaces the original `Arc<RwLock<LLMEngine>>` shared-state model with
6 actors + 4 GPU runner processes. The engine loop is a tokio task that
drives the `SchedD` actor via `sched.call(Step)`.

## Usage

The actor engine is the default runtime; there is no flag to enable it. The
operator-facing surface is unchanged from the pre-actor engine (the same CLI,
the same `/v1/*` endpoints). The differences are internal:

- **Startup**: `LLMEngine::new` spawns the 6 actors (the `FwdD`, the `SchedD`,
  the `StreamD`, the `HttpD`, the `KvD`, the `IpcChannel`) + the 4 GPU runner
  processes, then spawns the engine-loop tokio task. The loop drives
  `sched.call(Step)` repeatedly; when idle it sleeps 20ms.
- **Observability**: set `XINFER_DEBUG_ACTOR=1` to log every actor-boundary
  crossing (the forward result, the per-seq delivery, the cleanup). Default
  off (zero overhead).
- **Timeouts**: the GPU forward calls use `XINFER_FWD_TIMEOUT_MS` (default
  300s) so a slow prefill does not trip the actor RPC timeout. Control
  messages use the ractor default.
- **Shutdown**: dropping the `LLMEngine` sends `FwdMessage::Shutdown` to the
  `IpcChannel`, which broadcasts `MessageType::Shutdown` to the 4 runners +
  joins them. The actors are dropped with the engine.

## Interaction with pre-existing logic

The actor layer is a refactor of the existing engine, not a new feature. It
preserves the pre-existing behavior of:

- **The scheduler** (`Scheduler::schedule`/`postprocess`): unchanged logic;
  now owned by the `SchedD` actor instead of the engine's `RwLock`.
- **The runner** (`ModelRunner::run`): unchanged; now reached only via the
  `FwdD` → `IpcChannel` → runner-socket path (the sole socket writer).
- **The prefix cache + the block manager** (`BlockManager`): unchanged; now
  owned by the `KvD` actor.
- **The streaming delivery** (`StreamItem` channels): unchanged; now owned by
  the `StreamD` actor.

The one behavioral change is the elimination of the `RwLock<LLMEngine>`:
previously the server held the engine write-lock across the whole step; now
the server talks to the `HttpD`/`SchedD` actors and the lock is held only for
the brief state transitions inside each actor. This removes the lock
contention that serialized the HTTP ingestion against the inference loop.

## Academic Foundations

The actor model was introduced by Carl Hewitt in 1973 as a mathematical
framework for concurrent computation [1]. Hewitt's original formulation
defined actors as entities that:
- Have a unique name (address)
- Have an internal state
- Communicate exclusively via asynchronous message passing
- Process messages one at a time (sequential semantics)

The Erlang programming language (1986, Ericsson) [2] industrialized the
actor model with:
- Lightweight processes (actors) with mailboxes
- `let it crash` error handling (supervision trees)
- Hot code swapping
- Distributed transparency (actors can be on different nodes)

The `ractor` crate [3] implements the Erlang/OTP-style actor model in Rust
on top of tokio (or async-std). Key properties:
- Each actor is a tokio task with a mailbox (bounded or unbounded)
- Messages are processed one at a time (sequential semantics)
- Actors can supervise others (the OTP supervision tree)
- RPC via `call`/`call_t!` (the request/response with a per-call timeout)
- Fire-and-forget via `send_message`/`cast!`

xinfer uses ractor because the actor work is message-driven + the GPU
forward passes need a configurable per-call timeout (the `call` with a
custom `Duration`), which the prior `spawned-concurrency` framework
hardcoded to 5s (`DEFAULT_REQUEST_TIMEOUT`) and could not override
through the type-erased proxy.

### References

[1] C. Hewitt, "Actors and Messages: A New Model for Computation,"
    in *The Conceptual Basis of Programming Languages*, ACM, 1973.
    Also: P. A. Abdelsatif, "A Formalization of the Actor Model,"
    *Information Processing Letters*, 1985.

[2] J. Armstrong, "The Philosophy of Erlang," *ACM SIGPLAN Notices*,
    vol. 41, no. 3, 2006. Also: J. V. L. R. et al., "Erlang: A Language
    for Flexible and Distributed Software," *IEEE Computer*, 1996.

[3] `ractor` crate, v0.16.5, by slawlor.
    <https://github.com/slawlor/ractor>. The `#[ractor::actor]` focused
    macro generates the message enum + the `Actor` impl; the
    `#[ractor::async_trait]` attribute handles the async `handle` method.

## Actor Topology

```
                    LLMEngine (immutable facade, Arc<Self>)
                    holds
          |        |        \
          |        |         |
   ActorRef<FwdMessage> ActorRef<SchedMessage> ActorRef<StreamMessage> ActorRef<HttpMessage> ActorRef<KvMessage>
          |        |         |          |        |
          v        v         v          v        v
     +------+ +--------+ +--------+ +--------+ +----+
     | FwdD | | SchedD | |StreamD | | HttpD  | |KvD|
     +------+ +--------+ +--------+ +--------+ +----+
          |        |
          v        v
     +-------------+  +------------------+
     | IpcChannel  |  | 4 GPU runner     |
     | (sole sock  |  | processes        |
     |  writer)    |  | (NCCL TP group)  |
     +-------------+  +------------------+
```

Note: the `SmpD` actor was dropped during the Ractor conversion: it was dead
code, never wired into the pipeline. The sampling state lives in the runner
process, not in a CPU-side actor.

## Actor Responsibilities

| Actor | Runtime | Owns | Message enum |
|-------|---------|------|--------------|
| `SchedD` | tokio task | `Scheduler` | `SchedMessage` |
| `FwdD` | tokio task | `FwdTransport` (the `IpcChannel` ref) | `FwdMessage` |
| `IpcChannel` | tokio task (socket I/O on the blocking pool) | `Vec<LocalStream>` (4 runner sockets) | `IpcMessage` |
| `StreamD` | tokio task | `stream_senders`, `stream_decoders`, `request_types`, `decode_start_times`, `decode_length` | `StreamMessage` |
| `HttpD` | tokio task | `active_requests`, `cancelled_sequences`, `seq_prompt_replays` | `HttpMessage` |
| `KvD` | tokio task | `BlockManager` (block pool, prefix cache, mamba slots) | `KvMessage` |

## The Actor Invariant (Proven)

1. **Sole socket writer**: `IpcChannel` is the only code that touches runner
   sockets. `FwdD` drives it via the `ActorRef<IpcMessage>`. No other actor
   or engine method calls `send_local`/`receive_local`.

2. **No shared transport reference**: `LLMEngine` holds only actor refs
   (`ActorRef<FwdMessage>`, `ActorRef<SchedMessage>`, etc.). It does NOT
   hold a `Scheduler`, `BlockManager`, `RunnerType`, or raw socket.

3. **Actor-driven loop**: `start_engine` spawns a tokio task that calls
   `sched.call(Step)` in a loop. The `SchedD` actor drives `FwdD` +
   `StreamD` + `KvD` + `HttpD` internally. The engine task never calls the
   scheduler or the transport directly.

## Ractor-specific mechanics

### Blocking socket I/O offloaded to the blocking pool
The `IpcChannel` actor's `do_io` (the `send_local`/`receive_local` socket
I/O) is blocking. Since ractor actors are tokio tasks, blocking the task
would stall the runtime. The fix: the `IpcChannel` actor wraps the
socket I/O in `tokio::task::spawn_blocking`, so the blocking work runs on the
tokio blocking thread pool (not the async worker threads). The sockets are
shared with the blocking thread via an `Arc<Mutex<Vec<LocalStream>>>`.

### Per-call timeouts (the fix for the `RequestTimeout` panic)
The prior `spawned-concurrency` framework hardcoded a 5s
`DEFAULT_REQUEST_TIMEOUT` into every generated request method, unreachable
through the type-erased proxy. A GPU forward pass (a large prefill chunk)
legitimately takes >5s, so the `FwdD` `round_trip` timed out
+ the `.unwrap()` panicked, killing the actor chain.

Ractor's `ActorRef::call(msg_builder, timeout)` takes an explicit
`Option<Duration>` per call. The `FwdD` actor's GPU-forward calls
(`Run`, `RunMtp`, `RunDflash`, `RunSpecFf`, `Embed`) use a long timeout
(`XINFER_FWD_TIMEOUT_MS`, default 300s) so a slow GPU forward pass does not
time out. Control messages (the `Fire`, the `Shutdown`) use the default.
A timeout returns `CallResult::Timeout` (handled, not panicked).

### The engine loop is a tokio task
`start_engine` spawns a tokio task (via `GLOBAL_RT.spawn`) that loops on
`sched.call(Step)`. The loop sleeps (via `tokio::time::sleep`) when idle.
This replaces the prior native-thread engine loop.

## Message Flow: One Inference Step

```
engine-loop tokio task
  1. engine.is_idle()
       HttpD.call_idle()
       [if idle: sleep 20ms, continue]

  2. sched.call(Step)
       SchedD::handle_step:
       a. scheduler.schedule()
            KvD.can_allocate(seq)
            KvD.allocate(seq)
            [prefill: mamba fallback]
            [decode: KvD.may_append(seq)]
            [MTP: scheduler.pre_allocate_mtp_blocks]
b. fwd.run(owned_seqs)   [or fwd.run_mtp / fwd.run_dflash for the
           speculative-decode single-seq case]
             FwdD::handle_run:
               IpcChannel.round_trip(RunPrefill/RunDecode)
                 [spawn_blocking: the 4 GPU runners deserialize, forward, sample]
                 [rank-0 response is authoritative]
c. scheduler.postprocess(scheduled_ids, multi_output_ids)
             [append tokens, handle EOS/stop, mark finished]
        d. per-seq delivery loop:
            StreamD.get_request_type(seq_id)
            StreamD.decode_tokens(seq_id, token_ids)
              [returns Vec<(u32, String)> pairs]
            StreamD.deliver_token(seq_id, text, token_id)
              [tokio mpsc try_send to client]
            [if finished:]
              StreamD.deliver_done(seq_id, Done/Completion)
              StreamD.forget_sequence(seq_id)
              HttpD.remove_active_request(seq_id)
              FwdD.finished(seq_id)
                [IpcChannel.fire(FinishDecode)]
e. scheduler.clear_finished()
        f. check_canceled(reason)
            HttpD.take_cancellations()
            [for each cancelled seq:]
              Scheduler.cancel(seq_id)
              FwdD.finished(seq_id)
              HttpD.remove_active_request(seq_id)
              StreamD.forget_sequence(seq_id)
              StreamD.deliver_done(seq_id, Error)
            Scheduler.clear_finished()
            Scheduler.take_pending_runner_releases()
            FwdD.finished(seq_id) for each

  3. if task_processed == 0:
       engine.check_canceled(None)
       sleep 1ms (or PD cooling period)
```

## GPU Runner Broadcast (4-GPU Tensor Parallelism)

Each inference step broadcasts the same message to all 4 GPU runners via
`IpcChannel::do_io` (run on the tokio blocking pool):

```
IpcChannel::do_io(msg, expect_response=true):
  1. send_local(&mut streams, msg, false)
     // streams: Vec<LocalStream> (4 elements)
     // For each of 4 sockets:
     //   write_all(len.to_le_bytes())  // 4-byte LE length prefix
     //   write_all(rmp_serde::to_vec(msg))  // msgpack payload
  2. for each of 4 sockets:
     //   read_exact(len_buf[4])
     //   read_exact(msg_buf[len])
     //   rmp_serde::from_slice(msg_buf)
  3. return rank-0's response (authoritative)
     // ranks 1-3 are drained to maintain socket frame alignment
```

All 4 GPUs run the same model (tensor-parallel: each holds 1/N of the
attention heads + 1/N of the MLP experts). NCCL all-reduce combines the
shards; rank-0's output is authoritative.

## Circular Dependency Resolution

`SchedD` needs `ActorRef<HttpMessage>` (for `take_prompt_replay`, `remove_active_request`).
`HttpD` needs `ActorRef<SchedMessage>` (for `cancel` forwarding).

Resolution:
1. Create `SchedD` with `http: Mutex<Option<ActorRef<HttpMessage>>>` = `None`
2. Create `HttpD` with the `SchedD` ref
3. Call `SchedMessage::SetHttp(http_ref)` to wire the reference

The `Mutex<Option<ActorRef<HttpMessage>>>` is safe because:
- `HttpD.cancel` is a fire-and-forget `send_message`, never blocks
- `SchedD` only reads the `ActorRef<HttpMessage>` (never holds the lock across an await)

## Debug Logging

Set `XINFER_DEBUG_ACTOR=1` to enable verbose logging at every actor boundary:
- Forward pass result (sequence count, prefill/decode, output lengths)
- Per-sequence delivery entry (running index, sequence ID, finished flag, step tokens)
- Cleanup (clear_finished + check_canceled)

Default off (zero overhead).

## Key Design Decisions

1. **Tokio tasks, not native threads**: All actors run as tokio tasks (the
   ractor default runtime). The blocking socket I/O in `IpcChannel` is
   offloaded to the tokio blocking pool via `spawn_blocking`, so it never
   stalls the async worker threads.

2. **Per-call timeouts**: `ActorRef::call(msg, Some(duration))` takes
   an explicit timeout. The GPU forward passes use a long timeout
   (`XINFER_FWD_TIMEOUT_MS`, default 300s); control messages use the default.
   A timeout returns `CallResult::Timeout` (handled, not panicked).

3. **Sequence cloning at actor boundaries**: The actor framework requires
   owned payloads (no references across task boundaries). Sequences are
   cloned when crossing actor boundaries. This is the cost of the actor
   model. The `KvProtocol` mutating methods return the modified sequence
   to avoid losing block_table mutations.

4. **Rank-0 authoritative**: In multi-GPU tensor-parallel mode, all 4 runners
   compute the same result (NCCL all-reduce). Only rank-0's response is used;
   the others are drained to maintain socket frame alignment.

5. **`decode_tokens` returns `Vec<(u32, String)>`**: The decoder filters out
   EOS/special tokens (returns `None` for them). The delivery loop iterates
   over the pairs, so filtered tokens never reach the client. This prevents
   the EOS token from leaking into the chat output.

## Dependency Chain

The actor branch's design dependency is `ractor` (the actor framework). The
other dependencies below are pre-existing on the base (inherited, not introduced by
the actor sprint); they are listed for context on what the runner process
links against.

```
xinfer (this crate)
  |
  +-- ractor 0.16.5 (actor framework, the tokio-based) [int actor branch's dep]
  |     +-- the #[ractor::actor] focused macro (the message enum + the Actor impl)
  |     +-- the #[ractor::async_trait] (the async handle method)
  |     +-- ActorRef::call / call_t! (the per-call timeout RPC)
  |
  +-- llguidance (git, grammar FSM + PDA compilation) [pre-existing dep]
  |     +-- parser/ (Earley parser, CGrammar, PdaGrammar adapter)
  |     +-- toktrie/ (tokenizer trie, VOB bitsets)
  |     +-- toktrie_hf_tokenizers/ (HF tokenizer integration)
  |
  +-- pushdown-rs (PDA machine + CUDA package) [pre-existing dep]
  |     +-- machine.rs (PdaMachine: 7-tuple, CSR, SoA flat arrays)
  |     +-- compile.rs (RTN construction from CFG)
  |     +-- cuda.rs (CudaPackage: bitvec + CSR offsets for GPU upload)
  |     +-- mask_class.rs (VOB bitset construction from PDA mask)
  |
  +-- attention-rs (GPU kernels) [pre-existing dep]
        +-- fused sampler (top-k + top-p + VOB AND in one kernel)
        +-- PDA draft validation (GPU-side grammar walk)
        +-- NCCL tensor-parallel attention

The grammar, PDA, + VOB dependencies above are present in the dependency graph
but their features are wired on the guidance branch, which sits above this one.
On this branch they are inert (loaded but unused invoked).
```