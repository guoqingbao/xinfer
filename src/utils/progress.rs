use crate::runner::send_local;
use crate::runner::{receive_local, MessageType};
use candle_core::Result;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use interprocess::local_socket::traits::{Listener, Stream};
use interprocess::local_socket::{GenericNamespaced, ToNsName};
use interprocess::local_socket::{ListenerOptions, Stream as LocalStream};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::{thread, time};
pub trait ProgressLike: Send + Sync {
    /// Returns Ok with progress values, or Err if the connection is broken
    /// (e.g. runner process crashed during model loading).
    fn get_progress(&mut self) -> std::result::Result<Vec<(usize, usize)>, String>;
    fn set_progress(&mut self, p: usize);
}

pub struct ProgressReporter {
    pub rank: usize,
    pub progress: usize,
}

impl ProgressLike for ProgressReporter {
    fn get_progress(&mut self) -> std::result::Result<Vec<(usize, usize)>, String> {
        Ok(vec![(self.rank, self.progress)])
    }

    fn set_progress(&mut self, p: usize) {
        self.progress = p;
    }
}

impl ProgressReporter {
    pub fn new(rank: usize) -> Self {
        Self { rank, progress: 0 }
    }
}

unsafe impl Send for ProgressReporter {}
unsafe impl Sync for ProgressReporter {}

pub struct RemoteProgressReporter {
    pub rank: usize,
    pub progress: usize,
    pub streams: Vec<LocalStream>,
}

impl RemoteProgressReporter {
    pub fn new(rank: usize, shards: usize, sock_name: String, client: bool) -> Result<Self> {
        let mut streams = Vec::<LocalStream>::with_capacity(shards);
        if client {
            crate::log_info!("Remote progress reporter initialized for rank {}", rank);
            let name = sock_name.clone().to_ns_name::<GenericNamespaced>()?;
            let mut stream = LocalStream::connect(name.clone());

            loop {
                if stream.is_ok() {
                    break;
                }
                crate::log_info!("Runner retry connecting to socket: {}", sock_name);
                stream = LocalStream::connect(name.clone());
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            if let Ok(s) = stream {
                streams.push(s);
            } else {
                crate::log_error!("Failed to connect stream");
            }
        } else {
            let listener = ListenerOptions::new()
                .name(
                    sock_name
                        .clone()
                        .to_ns_name::<GenericNamespaced>()
                        .expect("Failed to to_ns_name"),
                )
                .create_sync()?;

            crate::log_info!("listener starting accepting runner {}", rank);
            for _ in 0..shards {
                match listener.accept() {
                    Ok(stream) => streams.push(stream),
                    Err(e) => {
                        crate::log_error!("Failed to accept connection: {}", e);
                    }
                }
            }
        }
        Ok(Self {
            rank,
            progress: 0,
            streams,
        })
    }
}

impl ProgressLike for RemoteProgressReporter {
    fn get_progress(&mut self) -> std::result::Result<Vec<(usize, usize)>, String> {
        let mut progress_values = Vec::with_capacity(self.streams.len());
        for mut stream in &mut self.streams {
            match receive_local(&mut stream, false) {
                Ok(MessageType::LoadingProgress((rank, progress))) => {
                    progress_values.push((rank, progress));
                }
                Ok(_) => {}
                Err(e) => {
                    return Err(format!(
                        "Runner process disconnected during model loading: {e}"
                    ));
                }
            }
        }
        Ok(progress_values)
    }

    fn set_progress(&mut self, p: usize) {
        let _ = send_local(
            &mut self.streams,
            &MessageType::LoadingProgress((self.rank, p)),
            false,
        );
    }
}

unsafe impl Send for RemoteProgressReporter {}
unsafe impl Sync for RemoteProgressReporter {}

pub struct Progress {
    m: MultiProgress,
    bars: Vec<ProgressBar>,
    size: usize,
}

impl Progress {
    pub fn new(n: usize, size: usize) -> Progress {
        let m = MultiProgress::new();
        let sty = ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:60.cyan/blue} {pos:>4}/{len:4} {msg}",
        )
        .unwrap()
        .progress_chars("##-");

        let mut bars = Vec::<ProgressBar>::new();
        for i in 0..n {
            let pb = m.add(ProgressBar::new(size as u64));
            pb.set_style(sty.clone());
            if n > 1 {
                pb.set_message(format!("On Rank {} Device", i));
            }
            bars.push(pb);
        }

        // if n > 1 {
        //     m.println(format!("Loading model in {} ranks!", n)).unwrap();
        // }
        Self { m, bars, size }
    }

    pub fn update(&self, idx: usize, progress: usize) {
        if idx < self.bars.len() && progress > 0 {
            let pos = self.bars[idx].position();
            self.bars[idx].inc(progress as u64 - pos);
            if self.bars.len() > 1 {
                if progress >= self.size {
                    self.bars[idx].set_message(format!("On Rank {} Device Finished", idx));
                } else {
                    self.bars[idx].set_message(format!("On Rank {} Device", idx));
                }
            }
        }
    }

    pub fn finish(&self) {
        for idx in 0..self.bars.len() {
            let pos = self.bars[idx].position();
            self.bars[idx].inc(self.size as u64 - pos);
            if self.bars.len() > 1 {
                self.bars[idx].set_message(format!("On Rank {} Device Finished", idx));
            }
        }
        let _ = self.m.clear();
    }
}

#[allow(unused_variables)]
pub fn progress_worker(
    num_shards: usize,
    length: usize,
    progress_reporter: &Arc<RwLock<Box<dyn ProgressLike>>>,
) -> std::thread::JoinHandle<()> {
    let mut finished_map = HashMap::<usize, usize>::new();
    let reporter = progress_reporter.clone();
    let progress_bar = Some(Progress::new(num_shards, length));
    let handle = thread::spawn(move || loop {
        {
            let _ = thread::sleep(time::Duration::from_millis(10 as u64));
            match reporter.write().get_progress() {
                Ok(progress) => {
                    for (rank, progress) in progress {
                        finished_map.insert(rank, progress);
                        if let Some(pb) = progress_bar.as_ref() {
                            pb.update(rank, progress);
                        }
                    }

                    if finished_map.values().all(|v| v >= &length) {
                        if let Some(pb) = progress_bar.as_ref() {
                            pb.finish();
                        }
                        break;
                    }
                }
                Err(e) => {
                    crate::log_error!("{}", e);
                    if let Some(pb) = progress_bar.as_ref() {
                        pb.finish();
                    }
                    break;
                }
            }
        }
    });

    handle
}

pub fn spawn_progress_thread(
    num_shards: usize,
    length: usize,
    progress_sock_name: String,
) -> JoinHandle<Option<JoinHandle<()>>> {
    thread::spawn(move || {
        match RemoteProgressReporter::new(0, num_shards, progress_sock_name, false) {
            Ok(reporter) => {
                let reporter: Arc<RwLock<Box<dyn ProgressLike>>> =
                    Arc::new(RwLock::new(Box::new(reporter)));

                // Call the real worker — assumed to return a JoinHandle
                let handle = progress_worker(num_shards, length, &reporter);
                Some(handle)
            }
            Err(e) => {
                eprintln!("Unable to create progress monitor: {e}");
                None
            }
        }
    })
}
