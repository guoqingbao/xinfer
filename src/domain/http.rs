//! The `HttpD` domain (the HTTP ingestion domain). It owns the per-request stream
//! channels + the request lifecycle state (the active requests, the cancellations,
//! the prompt-replay candidates) + handles the `Request`/`Cancel` messages (from
//! the HTTP handlers) + forwards the sequence to the `SchedD` domain.
//!
//! This is the domain that replaces the `LLMEngine`'s direct request handling
//! (the `add_request`/`add_request_pretokenized`). The `HttpD` actor owns the
//! request ingestion state + communicates with the other domains via messages
//! (the no shared locks).

use ractor::ActorProcessingErr;
use ractor::{Actor, ActorRef, RpcReplyPort};
use std::collections::{HashMap, HashSet};
use tokio::sync::mpsc::{channel, Receiver};

use crate::core::engine::StreamItem;
use crate::domain::sched::SchedMessage;
use crate::domain::stream::StreamMessage;

/// The HTTP ingestion actor.
pub struct HttpD;

/// The actor's state: the request lifecycle state + the refs to the other actors.
pub struct HttpState {
    /// The scheduler actor (the sequence is forwarded to).
    sched: ActorRef<SchedMessage>,
    /// The stream delivery actor (the stream channel is registered here).
    stream: ActorRef<StreamMessage>,
    /// The in-flight request ids (the engine is idle when this is empty).
    active_requests: HashSet<usize>,
    /// The sequences to cancel (drained by the engine loop).
    cancelled_sequences: Vec<usize>,
    /// The prompt-replay candidates (the seq_id -> the replay token ids).
    seq_prompt_replays: HashMap<usize, Vec<u32>>,
}

/// The `HttpD` actor's message enum (the explicit form).
pub enum HttpMessage {
    // The stream registration (the HTTP ingestion).
    RegisterStream(usize, RpcReplyPort<Receiver<StreamItem>>),
    Cancel(usize),
    // The request lifecycle state (owned by this actor).
    AddActiveRequest(usize),
    RemoveActiveRequest(usize),
    IsIdle(RpcReplyPort<bool>),
    MarkCancelled(usize),
    CancelAllActive,
    TakeCancellations(RpcReplyPort<Vec<usize>>),
    SetPromptReplay(usize, Vec<u32>),
    TakePromptReplay(usize, RpcReplyPort<Option<Vec<u32>>>),
    ClearPromptReplays,
}

#[ractor::async_trait]
impl ractor::Actor for HttpD {
    type Msg = HttpMessage;
    type State = HttpState;
    type Arguments = (ActorRef<SchedMessage>, ActorRef<StreamMessage>);

    async fn pre_start(
        &self,
        _myself: ActorRef<HttpMessage>,
        args: (ActorRef<SchedMessage>, ActorRef<StreamMessage>),
    ) -> Result<HttpState, ActorProcessingErr> {
        let (sched, stream) = args;
        Ok(HttpState {
            sched,
            stream,
            active_requests: HashSet::new(),
            cancelled_sequences: Vec::new(),
            seq_prompt_replays: HashMap::new(),
        })
    }

    async fn handle(
        &self,
        _myself: ActorRef<HttpMessage>,
        message: HttpMessage,
        state: &mut HttpState,
    ) -> Result<(), ActorProcessingErr> {
        use HttpMessage::*;
        match message {
            RegisterStream(seq_id, reply) => {
                let (tx, rx) = channel(1024);
                let _ = state.stream.send_message(StreamMessage::Register(seq_id, tx));
                let _ = reply.send(rx);
            }
            Cancel(seq_id) => {
                let _ = state.sched.send_message(SchedMessage::Cancel(seq_id));
            }
            AddActiveRequest(seq_id) => {
                state.active_requests.insert(seq_id);
            }
            RemoveActiveRequest(seq_id) => {
                state.active_requests.remove(&seq_id);
            }
            IsIdle(reply) => {
                let _ = reply.send(state.active_requests.is_empty());
            }
            MarkCancelled(seq_id) => {
                if !state.cancelled_sequences.contains(&seq_id) {
                    state.cancelled_sequences.push(seq_id);
                }
            }
            CancelAllActive => {
                for seq_id in state.active_requests.iter() {
                    state.cancelled_sequences.push(*seq_id);
                }
            }
            TakeCancellations(reply) => {
                let _ = reply.send(std::mem::take(&mut state.cancelled_sequences));
            }
            SetPromptReplay(seq_id, replay) => {
                state.seq_prompt_replays.insert(seq_id, replay);
            }
            TakePromptReplay(seq_id, reply) => {
                let _ = reply.send(state.seq_prompt_replays.remove(&seq_id));
            }
            ClearPromptReplays => {
                state.seq_prompt_replays.clear();
            }
        }
        Ok(())
    }
}

/// Start the `HttpD` actor (the HTTP ingestion domain) on the tokio runtime.
/// Returns the `ActorRef` (the only handle other actors may hold).
pub async fn start_http_actor(
    sched: ActorRef<SchedMessage>,
    stream: ActorRef<StreamMessage>,
) -> ActorRef<HttpMessage> {
    let (actor_ref, _) = HttpD::spawn(None, HttpD, (sched, stream))
        .await
        .expect("failed to start the HttpD actor");
    actor_ref
}