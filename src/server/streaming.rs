use super::ChatCompletionChunk;
use axum::response::sse::Event;
use futures::Stream;
use std::{
    pin::Pin,
    task::{Context, Poll},
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
