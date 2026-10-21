//! The `IpcChannel` actor (the sole socket writer). The ordered runner socket is
//! driven by exactly one actor, so it can never desync. Producers reach it only
//! through its `ActorRef<IpcMessage>`; no other code touches the socket directly.
//!
//! This is the transport for the `FwdD` actor when the model runs in separate
//! processes (the IPC runner). The in-process `ThreadRunner` case bypasses this
//! (the `FwdD` owns the `ModelRunner` directly).
//!
//! Ractor actors run as tokio tasks, so the blocking socket I/O
//! (`send_local`/`receive_local`) is offloaded to the blocking thread pool via
//! `tokio::task::spawn_blocking`. The runner sockets are shared with that
//! blocking thread through an `Arc<Mutex<Vec<LocalStream>>>`, so the sole-writer
//! invariant is preserved (only this actor + its blocking closure touch the
//! sockets, serialized by the mutex).

use interprocess::local_socket::Stream as LocalStream;
use ractor::ActorProcessingErr;
use ractor::{Actor, ActorRef, RpcReplyPort};
use std::sync::{Arc, Mutex};

use crate::runner::MessageType;

/// The `IpcChannel` actor's message enum (the explicit form, the no focused
/// macro).
pub enum IpcMessage {
    /// A request/response round trip: broadcast `msg` to every runner socket +
    /// drain every runner's response (the multi-rank frame-alignment invariant).
    /// Returns rank-0's response (authoritative; NCCL all-reduce makes all ranks
    /// compute the same result).
    RoundTrip(MessageType, RpcReplyPort<MessageType>),
    /// A fire-and-forget broadcast (the no response frame, e.g. `Finish`,
    /// `FinishDecode`, `SetLoopKick`).
    Fire(MessageType),
}

/// The `IpcChannel` actor (the sole socket writer).
pub struct IpcChannel;

/// The actor's state: the runner sockets, shared with the blocking I/O thread.
pub struct IpcState {
    streams: Arc<Mutex<Vec<LocalStream>>>,
}

#[ractor::async_trait]
impl ractor::Actor for IpcChannel {
    type Msg = IpcMessage;
    type State = IpcState;
    type Arguments = Arc<Mutex<Vec<LocalStream>>>;

    async fn pre_start(
        &self,
        _myself: ActorRef<IpcMessage>,
        streams: Arc<Mutex<Vec<LocalStream>>>,
    ) -> Result<IpcState, ActorProcessingErr> {
        Ok(IpcState { streams })
    }

    async fn handle(
        &self,
        _myself: ActorRef<IpcMessage>,
        message: IpcMessage,
        state: &mut IpcState,
    ) -> Result<(), ActorProcessingErr> {
        match message {
            IpcMessage::RoundTrip(msg, reply) => {
                let streams = state.streams.clone();
                let result = tokio::task::spawn_blocking(move || {
                    let mut guard = streams.lock().unwrap();
                    ipc_do_io(&mut guard, &msg, true)
                })
                .await
                .unwrap_or(MessageType::Error("IPC round_trip join failed".into()));
                let _ = reply.send(result);
            }
            IpcMessage::Fire(msg) => {
                let streams = state.streams.clone();
                let msg = msg.clone();
                let _ = tokio::task::spawn_blocking(move || {
                    let mut guard = streams.lock().unwrap();
                    ipc_do_io(&mut guard, &msg, false);
                })
                .await;
            }
        }
        Ok(())
    }
}

/// The actor's socket I/O (the sole writer): broadcast the request to all
/// runners + read the response from EVERY runner (the no socket is left
/// misframed). Rank-0's response is authoritative; the others are drained to
/// keep their sockets in frame alignment.
fn ipc_do_io(streams: &mut Vec<LocalStream>, msg: &MessageType, expect_response: bool) -> MessageType {
    if let Err(e) = crate::runner::send_local(streams, msg, false) {
        return MessageType::Error(format!("IPC send failed: {e}"));
    }
    if expect_response {
        let mut rank0: Option<MessageType> = None;
        for (i, stream) in streams.iter_mut().enumerate() {
            match crate::runner::receive_local(stream, false) {
                Ok(resp) => {
                    if i == 0 {
                        rank0 = Some(resp);
                    }
                    // The other ranks' responses are drained (the no socket is left
                    // misframed); rank-0 is authoritative.
                }
                Err(e) => {
                    return MessageType::Error(format!("IPC receive failed on rank {i}: {e}"));
                }
            }
        }
        rank0.unwrap_or_else(|| msg.clone())
    } else {
        msg.clone()
    }
}

/// Start the `IpcChannel` actor (the sole socket writer) on the tokio runtime.
/// Returns the `ActorRef` (the only handle other actors may hold).
pub async fn start_ipc_actor(streams: Vec<LocalStream>) -> ActorRef<IpcMessage> {
    let streams = Arc::new(Mutex::new(streams));
    let (actor_ref, _) = IpcChannel::spawn(None, IpcChannel, streams)
        .await
        .expect("failed to start the IpcChannel actor");
    actor_ref
}