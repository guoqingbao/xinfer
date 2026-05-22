use crate::runner::MessageType;
use interprocess::local_socket::traits::{Listener, Stream};
use interprocess::local_socket::{GenericNamespaced, Name, ToNsName};
use interprocess::local_socket::{ListenerOptions, Stream as LocalStream};
use std::fmt;
use std::io::Read;
use std::io::{BufRead, BufReader, Write};

pub struct CommandManager {
    daemon_streams: Option<Vec<LocalStream>>,
    main_stream: Option<LocalStream>,
}

impl fmt::Debug for CommandManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CommandManager")
            .field("daemon_streams", &self.daemon_streams)
            .field("main_stream", &self.main_stream)
            .finish()
    }
}

impl CommandManager {
    pub fn ipc_default_name() -> anyhow::Result<&'static str> {
        Ok("xinfer_daemon")
    }

    pub fn ipc_command_name(command_name: &str) -> anyhow::Result<String> {
        let printname = format!("command_{}", command_name);
        Ok(printname)
    }

    pub fn to_channel_name(name: &str) -> anyhow::Result<Name<'static>> {
        let printname = format!("{}.sock", name);
        Ok(printname.to_ns_name::<GenericNamespaced>()?)
    }

    //inter-node communication
    pub fn send_local(
        streams: &mut Vec<LocalStream>,
        message: &MessageType,
    ) -> std::io::Result<()> {
        let serialized = bincode::serialize(message).expect("Serialization failed");
        for stream in streams.iter_mut() {
            stream.write_all(&(serialized.len() as u32).to_le_bytes())?;
            stream.write_all(&serialized)?;
            stream.flush()?; // Ensure data is sent immediately
                             // Wait for acknowledgment
            let mut ack_buf = [0u8; 1];
            if let Err(e) = stream.read_exact(&mut ack_buf) {
                crate::log_info!(
                    "Timeout waiting for acknowledgment from subprocess: {:?}",
                    e
                );
            } else if ack_buf[0] != 1 {
                crate::log_info!("Unexpected acknowledgment value from subprocess");
            }
        }
        Ok(())
    }

    pub fn send_message(&mut self, message: &MessageType) -> std::io::Result<()> {
        assert!(self.daemon_streams.is_some(), "No daomon process found!");
        let streams = self.daemon_streams.as_mut().unwrap();
        CommandManager::send_local(streams, message)
    }

    pub fn receive_local(stream: &mut LocalStream) -> std::io::Result<MessageType> {
        let mut length_buf = [0u8; 4];
        stream.read_exact(&mut length_buf)?;
        let length = u32::from_le_bytes(length_buf) as usize;

        let mut serialized = vec![0u8; length];
        stream.read_exact(&mut serialized)?;
        let message: MessageType =
            bincode::deserialize(&serialized).expect("Deserialization failed");
        // Send acknowledgment
        stream.write_all(&[1])?;
        stream.flush()?;
        Ok(message)
    }

    pub fn receive_message(&mut self) -> std::io::Result<MessageType> {
        assert!(
            self.main_stream.is_some(),
            "not connected to the main process!"
        );
        CommandManager::receive_local(self.main_stream.as_mut().unwrap())
    }

    pub fn new_command(
        command_name: &str,
        num_subprocess: Option<usize>,
        is_daemon: bool,
    ) -> std::io::Result<Self> {
        let name = CommandManager::ipc_command_name(command_name).unwrap();
        CommandManager::new_channel(&name.as_str(), true, num_subprocess, is_daemon)
    }

    pub fn new_channel(
        channel_name: &str,
        is_command: bool,
        num_subprocess: Option<usize>,
        is_daemon: bool,
    ) -> std::io::Result<Self> {
        let sock_name = Self::to_channel_name(channel_name).unwrap();
        if is_daemon {
            crate::log_info!(
                "connect to main process' {} channel!",
                if is_command { "command" } else { "data" }
            );
            let mut stream = LocalStream::connect(sock_name)?;
            stream.write_all(b"ready\n")?;
            crate::log_warn!(
                "connected to the main process' {} channel!",
                if is_command { "command" } else { "data" }
            );
            Ok(Self {
                daemon_streams: None,
                main_stream: Some(stream),
            })
        } else {
            crate::log_info!(
                "build {} channel for the main process!",
                if is_command { "command" } else { "data" }
            );
            let num_subprocess = num_subprocess.unwrap();
            let listener = ListenerOptions::new().name(sock_name).create_sync()?;
            let mut streams = Vec::with_capacity(num_subprocess);
            for _ in 0..num_subprocess {
                let stream = listener.accept()?;
                crate::log_info!(
                    "accept one daemon process in {} channel!",
                    if is_command { "command" } else { "data" }
                );
                streams.push(stream);
            }

            for stream in streams.iter_mut() {
                let mut reader = BufReader::new(stream);
                let mut message = String::new();
                reader.read_line(&mut message)?;
                if message.trim() == "ready" {
                    crate::log_info!(
                        "one daemon process connected to the {} channel!",
                        if is_command { "command" } else { "data" }
                    );
                }
            }
            crate::log_warn!(
                "{} channel is built!",
                if is_command { "command" } else { "data" }
            );
            Ok(Self {
                daemon_streams: Some(streams),
                main_stream: None,
            })
        }
    }

    pub fn heartbeat(&mut self, is_daemon: bool) -> std::io::Result<()> {
        if is_daemon {
            match CommandManager::receive_local(self.main_stream.as_mut().unwrap()) {
                Ok(MessageType::Heartbeat) => Ok(()),
                Err(e) => Err(e),
                _ => Ok(()),
            }
        } else {
            let streams = self.daemon_streams.as_mut().unwrap();
            CommandManager::send_local(streams, &MessageType::Heartbeat)
        }
    }
}
