//! The `FwdD` actor (the forward-pass domain). It owns the model transport (the
//! `IpcChannel` actor, the sole socket writer) behind the uniform gate + handles
//! the inference ops (the `run`/`run_mtp`/`run_dflash`/`embed`) + the
//! fire-and-forget `finished`.
//!
//! This is the actor that replaces the `RunnerType` inference matches (the
//! `run_forward`/`run_mtp`/`run_dflash`/`run_spec_ff`): the transport
//! differentiation is hidden behind the `FwdD` actor, and the rest of the
//! pipeline just sends a message + receives a reply. No other actor holds a
//! direct transport reference (the `FwdD` ref is the only path to the model).
//!
//! Ractor actors run as tokio tasks. The GPU forward passes (the `run`/
//! `run_mtp`/`run_dflash`/`run_spec_ff` RPCs) can take far longer than the
//! framework reply timeout (a large prefill chunk), so each is called with an
//! explicit, long per-call timeout (`FWD_CALL_MS`, configurable via
//! `XINFER_FWD_TIMEOUT_MS`). A timeout returns `Err` (handled by the caller)
//! instead of panicking the actor.

use std::collections::HashMap;

use ractor::ActorProcessingErr;
use ractor::{Actor, ActorRef, RpcReplyPort};
use ractor::rpc::CallResult;

use crate::core::sequence::{DecodeSequence, Sequence};
use crate::domain::ipc::{IpcMessage, start_ipc_actor};
use crate::runner::MessageType;
use crate::server::EmbeddingStrategy;

/// The owned sequence batch (the mailbox-safe version of the borrowed `Seqs`).
#[derive(Debug, Clone)]
pub enum OwnedSeqs {
    /// A prefill batch (the full sequences).
    Prefill(Vec<Sequence>),
    /// A decode batch (the lightweight decode sequences).
    Decode(Vec<DecodeSequence>),
}

/// The forward-pass actor (the single-threaded actor that owns the model
/// transport behind the uniform gate).
pub struct FwdD;

/// The model's state: the transport (the `IpcChannel` actor ref, the sole
/// socket writer).
pub struct FwdState {
    ipc: ActorRef<IpcMessage>,
}

impl FwdState {
    /// A single IPC round trip with the long GPU-forward timeout. Returns the
    /// rank-0 response, or `None` on timeout / actor-stopped / send error.
    async fn ipc_round_trip(&self, msg: MessageType, timeout: std::time::Duration) -> Option<MessageType> {
        match self.ipc.call(|r| IpcMessage::RoundTrip(msg, r), Some(timeout)).await {
            Ok(CallResult::Success(m)) => Some(m),
            _ => None,
        }
    }
}

/// The per-call timeout for the GPU forward passes (a large prefill chunk can
/// take far longer than the default reply timeout). Configurable via
/// `XINFER_FWD_TIMEOUT_MS` (default 300s).
fn fwd_timeout_ms() -> u64 {
    std::env::var("XINFER_FWD_TIMEOUT_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(300_000)
}

/// The `FwdD` actor's message enum (the explicit form, the no focused macro).
pub enum FwdMessage {
    /// Run a prefill or decode step. Returns the sampled tokens.
    Run(OwnedSeqs, RpcReplyPort<Result<Vec<u32>, String>>),
    /// Run an MTP speculative decode step.
    RunMtp(Vec<DecodeSequence>, RpcReplyPort<Result<Vec<Vec<u32>>, String>>),
    /// Run a DFlash speculative decode step.
    RunDflash(Vec<DecodeSequence>, RpcReplyPort<Result<Vec<Vec<u32>>, String>>),
    /// Compute embeddings for the sequences.
    Embed(Vec<Sequence>, EmbeddingStrategy, RpcReplyPort<Result<Vec<Vec<f32>>, String>>),
    /// Notify the runner that a sequence has finished (the releases its slots).
    Finished(usize),
    /// The broadcast ops (the PD cache + the mamba prefix state).
    TrySwapKvcache(HashMap<usize, usize>, bool, RpcReplyPort<bool>),
    TrySendKvcache(Sequence, u32, RpcReplyPort<bool>),
    TryReceiveKvcache(Sequence, RpcReplyPort<(bool, u32, usize, usize)>),
    TryClearBlocks(Vec<u32>, RpcReplyPort<bool>),
    /// The PD transfer + the mamba prefix state ops (the broadcast ops).
    TryTransferPrefill(Sequence, RpcReplyPort<bool>),
    TryReceivePrefill(usize, RpcReplyPort<(bool, Option<Sequence>)>),
    TryCheckPrefillStatus(usize, RpcReplyPort<bool>),
    TryReleaseRemoteKvcache(usize, RpcReplyPort<bool>),
    TryCheckKvcacheRelease(usize, RpcReplyPort<bool>),
    TryCaptureMambaPrefixState(usize, u64, bool, RpcReplyPort<bool>),
    TryHasMambaPrefixState(u64, RpcReplyPort<bool>),
    TryRemoveMambaPrefixState(u64, RpcReplyPort<bool>),
    /// The raw message forwarding escape hatch (the multi-node worker forwards
    /// arbitrary `MessageType`s to the local runners). the sole socket writer).
    ForwardRaw(MessageType, RpcReplyPort<MessageType>),
    /// Shut down the transport (the fire-and-forget `Shutdown` broadcast to every
    /// runner socket, the clean teardown).
    Shutdown,
}

/// Build the `MessageType` for a `run` op (the prefill vs the decode).
fn run_msg(seqs: &OwnedSeqs) -> MessageType {
    match seqs {
        OwnedSeqs::Prefill(s) => MessageType::RunPrefill((s.clone(), true)),
        OwnedSeqs::Decode(d) => MessageType::RunDecode((d.clone(), false)),
    }
}

#[ractor::async_trait]
impl ractor::Actor for FwdD {
    type Msg = FwdMessage;
    type State = FwdState;
    type Arguments = ActorRef<IpcMessage>;

    async fn pre_start(
        &self,
        _myself: ActorRef<FwdMessage>,
        ipc: ActorRef<IpcMessage>,
    ) -> Result<FwdState, ActorProcessingErr> {
        Ok(FwdState { ipc })
    }

    async fn handle(
        &self,
        _myself: ActorRef<FwdMessage>,
        message: FwdMessage,
        state: &mut FwdState,
    ) -> Result<(), ActorProcessingErr> {
        use FwdMessage::*;
        let timeout = std::time::Duration::from_millis(fwd_timeout_ms());
        match message {
            Run(seqs, reply) => {
                let msg = run_msg(&seqs);
                let resp = state.ipc_round_trip(msg, timeout).await;
                let result = match resp {
                    Some(MessageType::RunResponse(ids)) if !ids.is_empty() => Ok(ids),
                    Some(MessageType::RunResponse(_)) => Err("runner returned an empty response".into()),
                    Some(MessageType::Error(e)) => Err(e),
                    Some(other) => Err(format!("unexpected IPC response: {other:?}")),
                    None => Err("IPC round_trip failed (timeout or actor stopped)".into()),
                };
                let _ = reply.send(result);
            }
            RunMtp(seqs, reply) => {
                let resp = state.ipc_round_trip(MessageType::RunDecodeMTP(seqs), timeout).await;
                let result = match resp {
                    Some(MessageType::RunResponseMTP(out)) => Ok(out),
                    Some(other) => Err(format!("unexpected IPC response: {other:?}")),
                    None => Err("IPC round_trip failed (timeout or actor stopped)".into()),
                };
                let _ = reply.send(result);
            }
            RunDflash(seqs, reply) => {
                let resp = state.ipc_round_trip(MessageType::RunDecodeDFlash(seqs), timeout).await;
                let result = match resp {
                    Some(MessageType::RunResponseDFlash(out)) => Ok(out),
                    Some(other) => Err(format!("unexpected IPC response: {other:?}")),
                    None => Err("IPC round_trip failed (timeout or actor stopped)".into()),
                };
                let _ = reply.send(result);
            }
            Embed(seqs, strategy, reply) => {
                let resp = state.ipc_round_trip(MessageType::RunEmbed((seqs, strategy)), timeout).await;
                let result = match resp {
                    Some(MessageType::RunResponseEmbed(out)) => Ok(out),
                    Some(other) => Err(format!("unexpected IPC response: {other:?}")),
                    None => Err("IPC round_trip failed (timeout or actor stopped)".into()),
                };
                let _ = reply.send(result);
            }
            Finished(id) => {
                let _ = state.ipc.send_message(IpcMessage::Fire(MessageType::FinishDecode(id)));
            }
            TrySwapKvcache(mappings, swap_in, reply) => {
                let resp = state.ipc_round_trip(MessageType::KVCacheSwap((mappings, swap_in)), timeout).await;
                let _ = reply.send(match resp {
                    Some(MessageType::KVCacheSwapResponse(v)) => v,
                    _ => false,
                });
            }
            TrySendKvcache(seq, token, reply) => {
                let resp = state.ipc_round_trip(MessageType::KvCacheSend((seq, token)), timeout).await;
                let _ = reply.send(match resp {
                    Some(MessageType::KvCacheSendResponse(v)) => v,
                    _ => false,
                });
            }
            TryReceiveKvcache(seq, reply) => {
                let resp = state.ipc_round_trip(MessageType::KvCacheReceive(seq), timeout).await;
                let _ = reply.send(match resp {
                    Some(MessageType::KvCacheReceiveResponse(v)) => v,
                    _ => (false, 0, 0, 0),
                });
            }
            TryClearBlocks(block_ids, reply) => {
                let resp = state.ipc_round_trip(MessageType::ClearBlocks(block_ids), timeout).await;
                let _ = reply.send(match resp {
                    Some(MessageType::ClearBlocksResponse(v)) => v,
                    _ => false,
                });
            }
            TryTransferPrefill(seq, reply) => {
                let resp = state.ipc_round_trip(MessageType::TransferPrefill(seq), timeout).await;
                let _ = reply.send(match resp {
                    Some(MessageType::TransferPrefillResponse(v)) => v,
                    _ => false,
                });
            }
            TryReceivePrefill(available_tokens, reply) => {
                let resp = state.ipc_round_trip(MessageType::ReceivePrefill(available_tokens), timeout).await;
                let _ = reply.send(match resp {
                    Some(MessageType::ReceivePrefillResponse(v)) => v,
                    _ => (false, None),
                });
            }
            TryCheckPrefillStatus(seq_id, reply) => {
                let resp = state.ipc_round_trip(MessageType::CheckPrefillStatus(seq_id), timeout).await;
                let _ = reply.send(match resp {
                    Some(MessageType::CheckPrefillStatusResponse(v)) => v,
                    _ => false,
                });
            }
            TryReleaseRemoteKvcache(seq_id, reply) => {
                let resp = state.ipc_round_trip(MessageType::KvCacheRelease(seq_id), timeout).await;
                let _ = reply.send(match resp {
                    Some(MessageType::KvCacheReleaseResponse(v)) => v,
                    _ => false,
                });
            }
            TryCheckKvcacheRelease(seq_id, reply) => {
                let resp = state.ipc_round_trip(MessageType::CheckKvCacheRelease(seq_id), timeout).await;
                let _ = reply.send(match resp {
                    Some(MessageType::CheckKvCacheReleaseResponse(v)) => v,
                    _ => false,
                });
            }
            TryCaptureMambaPrefixState(seq_id, hash, preserve, reply) => {
                let resp = state.ipc_round_trip(MessageType::CaptureMambaPrefixState((seq_id, hash, preserve)), timeout).await;
                let _ = reply.send(match resp {
                    Some(MessageType::CaptureMambaPrefixStateResponse(v)) => v,
                    _ => false,
                });
            }
            TryHasMambaPrefixState(hash, reply) => {
                let resp = state.ipc_round_trip(MessageType::HasMambaPrefixState(hash), timeout).await;
                let _ = reply.send(match resp {
                    Some(MessageType::HasMambaPrefixStateResponse(v)) => v,
                    _ => false,
                });
            }
            TryRemoveMambaPrefixState(hash, reply) => {
                let resp = state.ipc_round_trip(MessageType::RemoveMambaPrefixState(hash), timeout).await;
                let _ = reply.send(match resp {
                    Some(MessageType::RemoveMambaPrefixStateResponse(v)) => v,
                    _ => false,
                });
            }
            ForwardRaw(msg, reply) => {
                let resp = state.ipc_round_trip(msg, timeout).await;
                let _ = reply.send(resp.unwrap_or(MessageType::Error("forward_raw failed".into())));
            }
            Shutdown => {
                let _ = state.ipc.send_message(IpcMessage::Fire(MessageType::Shutdown));
            }
        }
        Ok(())
    }
}

/// Start the `IpcChannel` actor (the sole socket writer) + wrap it in the `FwdD`
/// actor (the forward-pass domain). Returns the `FwdD` ` ref (the only handle
/// other actors may hold).
pub async fn start_fwd_actor(streams: Vec<interprocess::local_socket::Stream>) -> ActorRef<FwdMessage> {
    let ipc = start_ipc_actor(streams).await;
    let (fwd_ref, _) = FwdD::spawn(None, FwdD, ipc)
        .await
        .expect("failed to start the FwdD actor");
    fwd_ref
}