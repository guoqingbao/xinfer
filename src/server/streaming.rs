use super::ChatCompletionChunk;
use axum::response::sse::Event;
use futures::Stream;
use std::{
    collections::VecDeque,
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
/// When disabled it is a lossless pass-through (holds at most one chunk, drained
/// immediately), so the caller has a single push/drain path regardless of the
/// switch - no parallel "immediate send" code path.
///
/// Sizing law (the "proportional inverse-derivative" pool):
///   base      = ceil(ema_rate * ema_gap * safety)   // bridge a typical stall
///   prebuffer = falling_rate * tau                  // the faster the rate is
///                                                     // dropping, the more we pre-buffer
///   capacity  = clamp(base + prebuffer, min_cap, max_cap)
const RES_PRIME_MS: u64 = 500;
/// Build-phase emit fraction of the sustained rate (a bit slower than production).
const RES_BUILD_FACTOR: f64 = 0.85;
/// Buffer floor: enough output to ride out 2s of no production.
const RES_BUFFER_TARGET_SECS: f64 = 2.0;

pub struct OutputReservoir {
    enabled: bool,
    /// Unbounded, lossless FIFO of produced-but-undelivered chunks.
    pool: VecDeque<(String, bool)>,
    /// Total chunks produced (drives the prime phase).
    total_pushed: usize,
    /// When the first chunk was produced (drives the prime phase).
    first_push: Option<Instant>,
    /// The recent production-rate window (the last 2s + the prior 2s, blended), so
    /// the emit rate tracks the current production instead of the lifetime average
    /// (which is dominated by the early burst).
    rate_window_start: Instant,
    rate_window_count: usize,
    rate_prev_count: usize,
    /// Fixed drain cadence.
    drain_interval: Duration,
    /// Sub-tick fractional emit remainder.
    emit_acc: f64,
}

impl OutputReservoir {
    pub fn new(enabled: bool) -> Self {
        let now = Instant::now();
        Self {
            enabled,
            pool: VecDeque::new(),
            total_pushed: 0,
            first_push: None,
            rate_window_start: now,
            rate_window_count: 0,
            rate_prev_count: 0,
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

    /// Sustained production rate (chunks/sec) over the recent window (the last 2s +
    /// the prior 2s, blended), so the emit rate tracks the current production
    /// instead of the lifetime average (which is dominated by the early burst).
    /// Falls back to the lifetime average when the recent window is too short.
    fn sustained_rate(&self, now: Instant) -> f64 {
        if self.total_pushed == 0 {
            return 0.0;
        }
        let window_elapsed = now.duration_since(self.rate_window_start).as_secs_f64();
        if window_elapsed < 0.5 {
            // The window is too short (just after a roll, or at startup): use the
            // lifetime average.
            if let Some(first) = self.first_push {
                return self.total_pushed as f64 / now.duration_since(first).as_secs_f64().max(1e-9);
            }
            return 0.0;
        }
        // The recent window: the current count + the prior count, over the total time.
        let total = self.rate_window_count + self.rate_prev_count;
        let total_time = window_elapsed + 2.0;
        total as f64 / total_time.max(1e-9)
    }

    /// Fill side: append a produced chunk. Lossless and unbounded when enabled;
    /// when disabled it holds at the newest chunk (pass-through).
    pub fn push(&mut self, text: String, is_reasoning: bool) {
        if !self.enabled {
            self.pool.push_back((text, is_reasoning));
            if self.pool.len() > 1 {
                self.pool.pop_front(); // keep only the newest for immediate delivery
            }
            return;
        }
        let now = Instant::now();
        if self.first_push.is_none() {
            self.first_push = Some(now);
        }
        self.total_pushed += 1;
        // Roll the rate window every 2s (the prior window becomes the blend partner).
        if now.duration_since(self.rate_window_start) >= Duration::from_secs(2) {
            self.rate_prev_count = self.rate_window_count;
            self.rate_window_count = 0;
            self.rate_window_start = now;
        }
        self.rate_window_count += 1;
        self.pool.push_back((text, is_reasoning));
    }

    /// Drain side: how many chunks to emit this tick.
    /// - Disabled: emit whatever is buffered (at most 1) immediately.
    /// - Enabled: prime (hold the first 0.5s), then build to ~2s of buffer at 85% of
    ///   the sustained rate, then maintain at 100%.
    pub fn drain_batch(&mut self) -> usize {
        if !self.enabled {
            return self.pool.len();
        }
        let now = Instant::now();
        let first = match self.first_push {
            Some(f) => f,
            None => return 0,
        };
        if now.duration_since(first) < Duration::from_millis(RES_PRIME_MS) {
            return 0;
        }
        let rate = self.sustained_rate(now);
        if rate <= 0.0 {
            return 0;
        }
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
}

#[cfg(test)]
mod reservoir_tests {
    use super::*;

    #[test]
    fn disabled_reservoir_is_a_pass_through() {
        let mut r = OutputReservoir::new(false);
        r.push("a".into(), false);
        r.push("b".into(), false); // only the newest is retained
        assert_eq!(r.drain_batch(), 1);
        assert_eq!(r.pop(), Some(("b".into(), false)));
        assert_eq!(r.drain_batch(), 0);
    }

    #[test]
    fn enabled_reservoir_primes_before_emitting() {
        let mut r = OutputReservoir::new(true);
        for i in 0..50 {
            r.push(format!("t{i}"), false);
        }
        // Within the 0.5s prime window nothing is emitted.
        assert_eq!(r.drain_batch(), 0);
        assert_eq!(r.pool.len(), 50);
    }

    #[test]
    fn enabled_reservoir_never_drops_chunks() {
        let mut r = OutputReservoir::new(true);
        for i in 0..100 {
            r.push(format!("t{i}"), false);
        }
        assert_eq!(r.pool.len(), 100); // lossless
    }
}
