use super::ChatCompletionChunk;
use axum::response::sse::Event;
use futures::Stream;
use std::collections::VecDeque;
use std::{
    pin::Pin,
    task::{Context, Poll},
    time::{Duration, Instant},
};
use tokio::sync::{mpsc, watch};

#[derive(PartialEq)]
pub enum StreamingStatus {
    Uninitialized,
    Started,
    Interrupted,
    Stopped,
}
pub enum ChatResponse {
    InternalError(String),
    ValidationError(String),
    ModelError(String),
    Chunk(ChatCompletionChunk),
    Done, //finish flag
}

pub struct Streamer {
    pub rx: mpsc::Receiver<ChatResponse>,
    pub status: StreamingStatus,
    pub disconnect_tx: Option<watch::Sender<bool>>,
}

impl Streamer {
    pub fn new(
        rx: mpsc::Receiver<ChatResponse>,
        disconnect_tx: Option<watch::Sender<bool>>,
    ) -> Self {
        Self {
            rx,
            status: StreamingStatus::Uninitialized,
            disconnect_tx,
        }
    }
}

impl Drop for Streamer {
    fn drop(&mut self) {
        if self.status != StreamingStatus::Stopped {
            if let Some(tx) = self.disconnect_tx.as_ref() {
                let _ = tx.send(true);
            }
        }
    }
}

impl Stream for Streamer {
    type Item = Result<Event, axum::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.status == StreamingStatus::Stopped {
            return Poll::Ready(None);
        }

        match self.rx.poll_recv(cx) {
            Poll::Ready(Some(resp)) => Poll::Ready(Some(self.get_mut().handle_response(resp))),
            Poll::Ready(None) => {
                if self.status == StreamingStatus::Started {
                    self.status = StreamingStatus::Interrupted;
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl Streamer {
    fn handle_response(&mut self, resp: ChatResponse) -> Result<Event, axum::Error> {
        match resp {
            ChatResponse::InternalError(e) => Ok(Event::default().data(e)),
            ChatResponse::ValidationError(e) => Ok(Event::default().data(e)),
            ChatResponse::ModelError(e) => Ok(Event::default().data(e)),
            ChatResponse::Chunk(response) => {
                if self.status != StreamingStatus::Started {
                    self.status = StreamingStatus::Started;
                }
                Event::default().json_data(response)
            }
            ChatResponse::Done => {
                self.status = StreamingStatus::Stopped;
                Ok(Event::default().data("[DONE]"))
            }
        }
    }
}

/// Per-stream output reservoir: decouples bursty token production from client
/// delivery by buffering produced chunks and draining them at a steady cadence
/// sized to the observed production rate. Smooths the staccato of alternating
/// prefill/decode steps across concurrent sequences.
///
/// Sizing law (the "proportional inverse-derivative" pool):
///   base      = ceil(ema_rate * ema_gap * safety)   // bridge a typical stall
///   prebuffer = falling_rate * tau                  // inverse-derivative: the
///                                                     // faster the rate is dropping,
///                                                     // the more we pre-buffer
///   capacity  = clamp(base + prebuffer, min_cap, max_cap)
/// A fast-output stream gets a bigger pool; a slow one a smaller pool.
/// Prime: hold back the first 0.5s of output before emitting.
const RES_PRIME_MS: u64 = 500;
/// Build-phase emit fraction of the sustained rate (a bit slower than production).
const RES_BUILD_FACTOR: f64 = 0.85;
/// Buffer floor: enough output to ride out 2s of no production.
const RES_BUFFER_TARGET_SECS: f64 = 2.0;

pub struct OutputReservoir {
    enabled: bool,
    /// Unbounded, lossless FIFO of produced-but-undelivered chunks.
    pool: VecDeque<(String, bool)>,
    /// Total chunks produced (drives the sustained production rate).
    total_pushed: usize,
    /// When the first chunk was produced (drives the sustained production rate).
    first_push: Option<Instant>,
    /// Fixed drain cadence.
    drain_interval: Duration,
    /// Sub-tick fractional emit remainder.
    emit_acc: f64,
}

impl OutputReservoir {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            pool: VecDeque::new(),
            total_pushed: 0,
            first_push: None,
            drain_interval: Duration::from_millis(10),
            emit_acc: 0.0,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// The fixed drain cadence (a `tokio::time::interval` period).
    pub fn drain_interval(&self) -> Duration {
        self.drain_interval
    }

    /// Sustained production rate (chunks/sec) = lifetime average. It falls off
    /// gradually during a stall (elapsed keeps growing), so the emit rate slows
    /// but never instantly stops.
    fn sustained_rate(&self, now: Instant) -> f64 {
        match self.first_push {
            Some(first) => self.total_pushed as f64 / now.duration_since(first).as_secs_f64().max(1e-9),
            None => 0.0,
        }
    }

    /// Fill side: append a produced chunk. Lossless and unbounded - nothing is
    /// ever dropped, no matter how fast the model produces.
    pub fn push(&mut self, text: String, is_reasoning: bool) {
        if !self.enabled {
            return;
        }
        let now = Instant::now();
        if self.first_push.is_none() {
            self.first_push = Some(now);
        }
        self.total_pushed += 1;
        self.pool.push_back((text, is_reasoning));
    }

    /// Drain side: how many chunks to emit this tick.
    /// - Prime: no emit for the first 0.5s (the buffer fills).
    /// - Build: emit at 85% of the sustained rate until the buffer holds ~2s.
    /// - Maintain: emit at 100% of the sustained rate (holds the buffer level).
    pub fn drain_batch(&mut self) -> usize {
        if !self.enabled {
            return 0;
        }
        let now = Instant::now();
        let first = match self.first_push {
            Some(f) => f,
            None => return 0,
        };
        // Prime: hold the first RES_PRIME_MS of output back before emitting.
        if now.duration_since(first) < Duration::from_millis(RES_PRIME_MS) {
            return 0;
        }
        let rate = self.sustained_rate(now);
        if rate <= 0.0 {
            return 0;
        }
        // RES_BUFFER_TARGET_SECS of output at the sustained rate is the buffer floor.
        let buffer_target = RES_BUFFER_TARGET_SECS * rate;
        let emit_rate = if (self.pool.len() as f64) < buffer_target {
            RES_BUILD_FACTOR * rate // build: a bit slower than production
        } else {
            rate // steady: maintain the buffer
        };
        self.emit_acc += emit_rate * self.drain_interval.as_secs_f64();
        let n = self.emit_acc as usize;
        self.emit_acc -= n as f64;
        n.min(self.pool.len())
    }

    /// Pop one chunk for delivery (drain side, FIFO order).
    pub fn pop(&mut self) -> Option<(String, bool)> {
        self.pool.pop_front()
    }

    /// Pending chunk count (for flush-on-end).
    pub fn pending(&self) -> usize {
        self.pool.len()
    }
}
