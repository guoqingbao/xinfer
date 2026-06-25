use super::ChatCompletionChunk;
use axum::response::sse::Event;
use flume::r#async::RecvFut;
use flume::Receiver;
use futures::Stream;
use std::future::Future;
use std::{
    pin::Pin,
    task::{Context, Poll},
};
use tokio::sync::watch;

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
    pub rx: Receiver<ChatResponse>,
    pub status: StreamingStatus,
    pub disconnect_tx: Option<watch::Sender<bool>>,
    recv_fut: Option<RecvFut<'static, ChatResponse>>,
    rx_static: Option<&'static Receiver<ChatResponse>>,
}

impl Streamer {
    pub fn new(
        rx: Receiver<ChatResponse>,
        disconnect_tx: Option<watch::Sender<bool>>,
    ) -> Self {
        Self {
            rx,
            status: StreamingStatus::Uninitialized,
            disconnect_tx,
            recv_fut: None,
            rx_static: None,
        }
    }
}

impl Drop for Streamer {
    fn drop(&mut self) {
        self.recv_fut = None;
        self.rx_static = None;
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

        // First try a non-blocking recv for immediate data
        match self.rx.try_recv() {
            Ok(resp) => return Poll::Ready(Some(self.get_mut().handle_response(resp))),
            Err(flume::TryRecvError::Disconnected) => {
                if self.status == StreamingStatus::Started {
                    self.status = StreamingStatus::Interrupted;
                    return Poll::Ready(None);
                }
                return Poll::Ready(None);
            }
            Err(flume::TryRecvError::Empty) => {}
        }

        // Register waker via recv_async for proper async notification
        if self.recv_fut.is_none() {
            let rx_ref: &'static Receiver<ChatResponse> =
                unsafe { &*(&self.rx as *const Receiver<ChatResponse>) };
            self.recv_fut = Some(rx_ref.recv_async());
            self.rx_static = Some(rx_ref);
        }

        if let Some(fut) = self.recv_fut.as_mut() {
            let pinned = unsafe { Pin::new_unchecked(fut) };
            match pinned.poll(cx) {
                Poll::Ready(Ok(resp)) => {
                    self.recv_fut = None;
                    Poll::Ready(Some(self.get_mut().handle_response(resp)))
                }
                Poll::Ready(Err(_)) => {
                    self.recv_fut = None;
                    if self.status == StreamingStatus::Started {
                        self.status = StreamingStatus::Interrupted;
                    }
                    Poll::Ready(None)
                }
                Poll::Pending => Poll::Pending,
            }
        } else {
            Poll::Pending
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
