//! The `StreamD` domain (the stream delivery domain). It owns ALL the per-request
//! delivery state (the stream channels, the decoders, the request types, the
//! decode timing) + handles the token delivery. This is the single owner of the
//! per-request delivery state (the no shared `stream_senders`/`stream_decoders`/
//! `request_types` on the engine).
//!
//! The `StreamD` actor is reached only through its `ActorRef<StreamMessage>`; no
//! other code holds the delivery state or mutates it directly.

use ractor::ActorProcessingErr;
use ractor::{Actor, ActorRef, RpcReplyPort};
use std::collections::HashMap;

use crate::core::engine::{RequestType, StreamItem};
use crate::core::DecodeStreamType;
use tokio::sync::mpsc::Sender;

/// The stream delivery actor.
pub struct StreamD;

/// The actor's state: all the per-request delivery state.
pub struct StreamState {
    /// The per-request stream channels.
    stream_senders: HashMap<usize, Sender<StreamItem>>,
    /// The per-request stream decoders (the token -> text conversion).
    stream_decoders: HashMap<usize, DecodeStreamType>,
    /// The per-request type (the Stream vs the Completion).
    request_types: HashMap<usize, RequestType>,
    /// The per-request decode start time.
    decode_start_times: HashMap<usize, usize>,
    /// The per-request decode length (the number of tokens decoded so far).
    decode_length: HashMap<usize, usize>,
    /// The per-request prefilled reasoning-end marker.
    seq_prefilled_reasoning_end: HashMap<usize, String>,
    /// The last throughput-log time (the 10s gate).
    last_check_throughput_time: usize,
}

/// The `StreamD` actor's message enum (the explicit form).
pub enum StreamMessage {
    // The registration (the HTTP ingestion).
    Register(usize, Sender<StreamItem>),
    SetDecoder(usize, DecodeStreamType),
    SetRequestType(usize, RequestType),
    RecordDecodeStart(usize, usize),
    RecordPrefilledReasoningEnd(usize, String),
    // The delivery (the sampler output). Fire-and-forget (the no reply needed;
    // the StreamD actor handles the delivery + the client channel).
    DeliverToken(usize, String, u32),
    DeliverTokenId(usize, u32),
    DeliverDone(usize, StreamItem),
    // The queries (the read-only accessors for the engine's delivery logic).
    GetRequestType(usize, RpcReplyPort<Option<RequestType>>),
    GetDecodeStartTime(usize, RpcReplyPort<Option<usize>>),
    GetDecodeLength(usize, RpcReplyPort<Option<usize>>),
    GetPrefilledReasoningEnd(usize, RpcReplyPort<Option<String>>),
    DecodeTokens(usize, Vec<u32>, RpcReplyPort<Vec<(u32, String)>>),
    SetDecodeLength(usize, usize),
    ForgetSequence(usize),
    ThroughputGate(usize, RpcReplyPort<bool>),
    ResetAll,
}

#[ractor::async_trait]
impl ractor::Actor for StreamD {
    type Msg = StreamMessage;
    type State = StreamState;
    type Arguments = ();

    async fn pre_start(
        &self,
        _myself: ActorRef<StreamMessage>,
        _args: (),
    ) -> Result<StreamState, ActorProcessingErr> {
        Ok(StreamState {
            stream_senders: HashMap::new(),
            stream_decoders: HashMap::new(),
            request_types: HashMap::new(),
            decode_start_times: HashMap::new(),
            decode_length: HashMap::new(),
            seq_prefilled_reasoning_end: HashMap::new(),
            last_check_throughput_time: 0,
        })
    }

    async fn handle(
        &self,
        _myself: ActorRef<StreamMessage>,
        message: StreamMessage,
        state: &mut StreamState,
    ) -> Result<(), ActorProcessingErr> {
        use StreamMessage::*;
        match message {
            Register(seq_id, tx) => {
                state.stream_senders.insert(seq_id, tx);
            }
            SetDecoder(seq_id, decoder) => {
                state.stream_decoders.insert(seq_id, decoder);
            }
            SetRequestType(seq_id, request_type) => {
                state.request_types.insert(seq_id, request_type);
            }
            RecordDecodeStart(seq_id, time) => {
                state.decode_start_times.insert(seq_id, time);
            }
            RecordPrefilledReasoningEnd(seq_id, marker) => {
                state.seq_prefilled_reasoning_end.insert(seq_id, marker);
            }
            DeliverToken(seq_id, text, token) => {
                if let Some(sender) = state.stream_senders.get_mut(&seq_id) {
                    let _ = sender.try_send(StreamItem::Token(text, token));
                }
            }
            DeliverTokenId(seq_id, token) => {
                if let Some(sender) = state.stream_senders.get_mut(&seq_id) {
                    let _ = sender.try_send(StreamItem::TokenID(token));
                }
            }
            DeliverDone(seq_id, item) => {
                if let Some(sender) = state.stream_senders.remove(&seq_id) {
                    let _ = sender.try_send(item);
                }
            }
            GetRequestType(seq_id, reply) => {
                let _ = reply.send(state.request_types.get(&seq_id).cloned());
            }
            GetDecodeStartTime(seq_id, reply) => {
                let _ = reply.send(state.decode_start_times.get(&seq_id).copied());
            }
            GetDecodeLength(seq_id, reply) => {
                let _ = reply.send(state.decode_length.get(&seq_id).copied());
            }
            GetPrefilledReasoningEnd(seq_id, reply) => {
                let _ = reply.send(state.seq_prefilled_reasoning_end.get(&seq_id).cloned());
            }
            DecodeTokens(seq_id, tokens, reply) => {
                let decoded = if let Some(decoder) = state.stream_decoders.get_mut(&seq_id) {
                    tokens
                        .iter()
                        .filter_map(|&t| decoder.step(t).ok().flatten().map(|text| (t, text)))
                        .collect()
                } else {
                    Vec::new()
                };
                let _ = reply.send(decoded);
            }
            SetDecodeLength(seq_id, n) => {
                state.decode_length.insert(seq_id, n);
            }
            ForgetSequence(seq_id) => {
                state.stream_senders.remove(&seq_id);
                state.stream_decoders.remove(&seq_id);
                state.request_types.remove(&seq_id);
                state.decode_start_times.remove(&seq_id);
                state.decode_length.remove(&seq_id);
                state.seq_prefilled_reasoning_end.remove(&seq_id);
            }
            ThroughputGate(cur_time_ms, reply) => {
                let passed = if cur_time_ms - state.last_check_throughput_time < 10000 {
                    false
                } else {
                    state.last_check_throughput_time = cur_time_ms;
                    true
                };
                let _ = reply.send(passed);
            }
            ResetAll => {
                state.stream_senders.clear();
                state.stream_decoders.clear();
                state.request_types.clear();
                state.decode_start_times.clear();
                state.decode_length.clear();
                state.seq_prefilled_reasoning_end.clear();
            }
        }
        Ok(())
    }
}

/// Start the `StreamD` actor (the stream delivery domain) on the tokio runtime.
/// Returns the `ActorRef` (the only handle other actors may hold).
pub async fn start_stream_actor() -> ActorRef<StreamMessage> {
    let (actor_ref, _) = StreamD::spawn(None, StreamD, ())
        .await
        .expect("failed to start the StreamD actor");
    actor_ref
}