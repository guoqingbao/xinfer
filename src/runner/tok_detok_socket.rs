// Length-prefixed bincode framing over `interprocess::local_socket::Stream`.
// Mirrors the `send_local`/`receive_local` pattern in `src/runner/mod.rs` so the
// engine↔tok_detok_worker channel uses the same transport class as engine↔runner.
//
// Wire format: [len: u32 LE][kind: u8][bincode(payload): len bytes]

use std::io::{BufReader, Read, Write};
use std::sync::{Arc, Mutex};

use interprocess::local_socket::traits::{Listener, Stream as StreamTrait};
use interprocess::local_socket::{
    GenericNamespaced, ListenerOptions, Stream as LocalStream, ToNsName,
};

use crate::runner::tok_detok_msgs::MsgKind;

#[inline]
fn build_frame(data: &[u8], kind: MsgKind) -> Vec<u8> {
    let mut frame = Vec::with_capacity(4 + 1 + data.len());
    frame.extend_from_slice(&(data.len() as u32).to_le_bytes());
    frame.push(kind as u8);
    frame.extend_from_slice(data);
    frame
}

/// Pair of engine-side endpoints — one for tokenize, one for detokenize.
/// Both connect to the same tok_detok_worker process; the worker runs them
/// on separate threads so the two directions don't head-of-line each other.
#[derive(Clone)]
pub struct TokDetokIpcPair {
    pub tok: TokDetokSocketServer,
    pub det: TokDetokSocketServer,
}

/// Engine-side endpoint. Accepts a single tok_detok_worker connection.
#[derive(Clone)]
pub struct TokDetokSocketServer {
    pub writer: Arc<Mutex<LocalStream>>,
    pub reader: Arc<Mutex<BufReader<LocalStream>>>,
}

impl TokDetokSocketServer {
    pub fn bind_and_accept(name: &str) -> std::io::Result<Self> {
        let ns_name = name.to_ns_name::<GenericNamespaced>()?;
        let listener = ListenerOptions::new().name(ns_name).create_sync()?;
        let stream = listener.accept()?;
        let reader_stream = interprocess::TryClone::try_clone(&stream)?;
        Ok(Self {
            writer: Arc::new(Mutex::new(stream)),
            reader: Arc::new(Mutex::new(BufReader::with_capacity(
                64 * 1024,
                reader_stream,
            ))),
        })
    }

    #[inline]
    pub fn send(&self, data: &[u8], kind: MsgKind) {
        let frame = build_frame(data, kind);
        let mut w = self.writer.lock().unwrap();
        let _ = w.write_all(&frame);
    }

    /// Returns `Ok(None)` on clean peer disconnect (the worker process exit
    /// closes its half of the socket pair). All other I/O errors bubble.
    pub fn recv_blocking(&self) -> std::io::Result<Option<(u8, Vec<u8>)>> {
        let mut r = self.reader.lock().unwrap();
        let mut len_buf = [0u8; 4];
        match r.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e),
        }
        let len = u32::from_le_bytes(len_buf) as usize;
        let mut kind_buf = [0u8; 1];
        r.read_exact(&mut kind_buf)?;
        let mut data = vec![0u8; len];
        r.read_exact(&mut data)?;
        Ok(Some((kind_buf[0], data)))
    }
}

/// Worker-side endpoint. Connects to the engine's listener.
pub struct TokDetokSocketClient {
    pub writer: Arc<Mutex<LocalStream>>,
    pub reader: Arc<Mutex<BufReader<LocalStream>>>,
}

impl TokDetokSocketClient {
    pub fn connect(name: &str) -> std::io::Result<Self> {
        let ns_name = name.to_ns_name::<GenericNamespaced>()?;
        for _ in 0..600 {
            if let Ok(stream) = LocalStream::connect(ns_name.clone()) {
                let reader_stream = interprocess::TryClone::try_clone(&stream)?;
                return Ok(Self {
                    writer: Arc::new(Mutex::new(stream)),
                    reader: Arc::new(Mutex::new(BufReader::with_capacity(
                        64 * 1024,
                        reader_stream,
                    ))),
                });
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        Err(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            "tok_detok_socket connect timeout",
        ))
    }

    #[inline]
    pub fn send(&self, data: &[u8], kind: MsgKind) {
        let frame = build_frame(data, kind);
        let mut w = self.writer.lock().unwrap();
        let _ = w.write_all(&frame);
    }

    /// Returns `Ok(None)` on clean peer disconnect (`UnexpectedEof` on the
    /// length-prefix read). Engine shutdown closes the socket, so the
    /// worker's recv loop must distinguish that from a real I/O error.
    /// All other read failures (truncated frame, transport error) bubble
    /// up as `Err`.
    pub fn recv_blocking(&self) -> std::io::Result<Option<(u8, Vec<u8>)>> {
        let mut r = self.reader.lock().unwrap();
        let mut len_buf = [0u8; 4];
        match r.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e),
        }
        let len = u32::from_le_bytes(len_buf) as usize;
        let mut kind_buf = [0u8; 1];
        r.read_exact(&mut kind_buf)?;
        let mut data = vec![0u8; len];
        r.read_exact(&mut data)?;
        Ok(Some((kind_buf[0], data)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::tok_detok_msgs::MsgKind;

    // End-to-end framing test: server and client are both in-process threads
    // sharing the same `interprocess::local_socket` channel used in production.
    #[test]
    fn frame_roundtrip_server_client() {
        let name = format!("xinfer-tokdetok-test-{}", std::process::id());
        let name_for_server = name.clone();

        let server_thread = std::thread::spawn(move || {
            TokDetokSocketServer::bind_and_accept(&name_for_server).expect("server bind")
        });

        let client = TokDetokSocketClient::connect(&name).expect("client connect");
        let server = server_thread.join().expect("server thread join");

        // server -> client
        server.send(b"hello", MsgKind::TokDetokInit);
        let (k, d) = client.recv_blocking().expect("recv").expect("frame");
        assert_eq!(k, MsgKind::TokDetokInit as u8);
        assert_eq!(&d, b"hello");

        // client -> server
        client.send(b"world!", MsgKind::DetokenizeResp);
        let (k, d) = server.recv_blocking().expect("recv").expect("frame");
        assert_eq!(k, MsgKind::DetokenizeResp as u8);
        assert_eq!(&d, b"world!");
    }

    // Empty payload must still frame correctly (len=0).
    #[test]
    fn frame_roundtrip_empty_payload() {
        let name = format!("xinfer-tokdetok-test-empty-{}", std::process::id());
        let name_for_server = name.clone();

        let server_thread = std::thread::spawn(move || {
            TokDetokSocketServer::bind_and_accept(&name_for_server).expect("server bind")
        });

        let client = TokDetokSocketClient::connect(&name).expect("client connect");
        let server = server_thread.join().expect("server thread join");

        server.send(b"", MsgKind::Error);
        let (k, d) = client.recv_blocking().expect("recv").expect("frame");
        assert_eq!(k, MsgKind::Error as u8);
        assert!(d.is_empty());
    }
}
