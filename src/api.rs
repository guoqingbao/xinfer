use crate::core::engine::{LLMEngine, StreamItem, GLOBAL_RT};
use crate::core::GenerationOutput;
use crate::server::{build_messages_and_images, run_server, ChatMessage};
use crate::tools::Tool;
use crate::utils::chat_template::Message;
use crate::utils::config::{EngineConfig, SamplingParams};
use crate::utils::get_dtype;
use candle_core::{DType, Result};
use parking_lot::RwLock;
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::mpsc;

#[derive(Clone, Debug)]
pub enum ModelRepo {
    /// (model_id, filename) -- when filename is None, treat as safetensor model id.
    /// When filename is Some, treat as GGUF model id + GGUF filename.
    ModelID((&'static str, Option<&'static str>)),
    /// Safetensor local path.
    ModelPath(&'static str),
    /// GGUF file(s). Only the first file is used today.
    ModelFile(Vec<&'static str>),
}

#[derive(Clone, Debug)]
pub struct EngineBuilder {
    repo: ModelRepo,
    isq: Option<String>,
    dtype: Option<DType>,
    flash_attn: Option<bool>,
    kvcache_dtype: Option<crate::utils::config::KvCacheDtype>,
    mamba_fraction: Option<f32>,
    prefix_cache: Option<bool>,
    prefix_cache_max_tokens: Option<usize>,
    pd_server_prefix_cache_ratio: Option<f32>,
    pd_client_prefix_cache_ratio: Option<f32>,
    yarn_scaling_factor: Option<f64>,
    device_ids: Option<Vec<usize>>,
}

impl EngineBuilder {
    pub fn new(repo: ModelRepo) -> Self {
        Self {
            repo,
            isq: None,
            dtype: None,
            flash_attn: None,
            kvcache_dtype: None,
            mamba_fraction: None,
            prefix_cache: None,
            prefix_cache_max_tokens: None,
            pd_server_prefix_cache_ratio: None,
            pd_client_prefix_cache_ratio: None,
            yarn_scaling_factor: None,
            device_ids: None,
        }
    }

    pub fn with_isq(mut self, isq: impl Into<String>) -> Self {
        self.isq = Some(isq.into());
        self
    }

    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }

    pub fn without_flash_attn(mut self) -> Self {
        self.flash_attn = Some(false);
        self
    }

    pub fn with_fp8_kvcache(mut self) -> Self {
        self.kvcache_dtype = Some(crate::utils::config::KvCacheDtype::Fp8);
        self
    }

    pub fn with_kvcache_dtype(mut self, dtype: crate::utils::config::KvCacheDtype) -> Self {
        self.kvcache_dtype = Some(dtype);
        self
    }

    pub fn with_mamba_fraction(mut self, ratio: f32) -> Self {
        self.mamba_fraction = Some(ratio);
        self
    }

    pub fn with_prefix_cache(mut self, enabled: bool) -> Self {
        self.prefix_cache = Some(enabled);
        self
    }

    pub fn with_prefix_cache_max_tokens(mut self, max_tokens: usize) -> Self {
        self.prefix_cache_max_tokens = Some(max_tokens);
        self
    }

    pub fn with_pd_server_prefix_cache_ratio(mut self, ratio: f32) -> Self {
        self.pd_server_prefix_cache_ratio = Some(ratio);
        self
    }

    pub fn with_pd_client_prefix_cache_ratio(mut self, ratio: f32) -> Self {
        self.pd_client_prefix_cache_ratio = Some(ratio);
        self
    }

    pub fn with_yarn_scaling_factor(mut self, factor: f64) -> Self {
        self.yarn_scaling_factor = Some(factor);
        self
    }

    pub fn with_multirank(mut self, device_ids: &str) -> Result<Self> {
        self.device_ids = Some(parse_device_ids(device_ids)?);
        Ok(self)
    }

    pub fn build(self) -> Result<Engine> {
        let (model_id, weight_path, weight_file) = match self.repo {
            ModelRepo::ModelID((model_id, filename)) => (
                Some(model_id.to_owned()),
                None,
                filename.map(|f| f.to_owned()),
            ),
            ModelRepo::ModelPath(path) => (None, Some(path.to_owned()), None),
            ModelRepo::ModelFile(files) => {
                if files.len() > 1 {
                    crate::log_warn!("Multiple GGUF files provided, using the first one.");
                }
                let weight_file = files.into_iter().next().map(|f| f.to_owned());
                (None, None, weight_file)
            }
        };

        let mut econfig = EngineConfig::new(
            model_id,
            weight_path,
            weight_file,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            self.isq,
            Some(self.device_ids.clone().unwrap_or(vec![0]).len()),
            self.device_ids.clone(),
            None,
            None,
            !self.prefix_cache.unwrap_or(true),
            self.prefix_cache_max_tokens,
            None,
            None,
            None,
            None,
            self.mamba_fraction,
            None,
            None,
            None,
            None,
            None,
            self.pd_server_prefix_cache_ratio,
            self.pd_client_prefix_cache_ratio,
            self.yarn_scaling_factor,
            false,
            false,
            None,
        );

        if let Some(kv_dtype) = self.kvcache_dtype {
            econfig.kvcache_dtype = kv_dtype;
        }

        let dtype = self.dtype.clone().map(dtype_to_str);
        let dtype = get_dtype(dtype);

        let engine = LLMEngine::new(&econfig, dtype)?;
        Ok(Engine { engine, econfig })
    }
}

pub struct Engine {
    engine: Arc<RwLock<LLMEngine>>,
    econfig: EngineConfig,
}

impl Engine {
    pub fn start_server(&mut self, port: usize, with_ui_server: bool) -> Result<()> {
        GLOBAL_RT.block_on(async {
            run_server(
                self.engine.clone(),
                self.econfig.clone(),
                port,
                with_ui_server,
            )
            .await
        })
    }

    pub fn generate(
        &mut self,
        params: SamplingParams,
        messages: Vec<ChatMessage>,
        tools: Vec<Tool>,
    ) -> Result<GenerationOutput> {
        let img_cfg = { self.engine.read().img_cfg.clone() };
        let (messages, image_data) = build_messages_and_images(&messages, img_cfg.as_ref())?;
        self.generate_messages(params, messages, image_data, tools)
    }

    pub fn generate_messages(
        &mut self,
        params: SamplingParams,
        messages: Vec<Message>,
        images: Option<crate::utils::image::ImageData>,
        tools: Vec<Tool>,
    ) -> Result<GenerationOutput> {
        let (receivers, tokenizer) = {
            let mut engine = self.engine.write();
            (
                engine.generate_sync(&vec![params], &vec![messages], images, &tools, &None)?,
                Arc::new(engine.tokenizer.clone()),
            )
        };

        let results = GLOBAL_RT.block_on(async {
            LLMEngine::collect_sync_results(receivers, tokenizer, None).await
        })?;

        // Extract GenerationOutput
        for result in results {
            return Ok(result);
        }

        candle_core::bail!("No generation output returned")
    }

    pub fn generate_stream(
        &mut self,
        params: SamplingParams,
        messages: Vec<ChatMessage>,
        tools: Vec<Tool>,
    ) -> Result<EngineStream> {
        let img_cfg = { self.engine.read().img_cfg.clone() };
        let (messages, image_data) = build_messages_and_images(&messages, img_cfg.as_ref())?;

        let (seq_id, prompt_length, _prefilled_reasoning_end, stream) = {
            let mut engine = self.engine.write();
            engine.generate_stream(&params, &messages, image_data, &tools, &None)?
        };

        Ok(EngineStream {
            engine: self.engine.clone(),
            rx: stream,
            finished: false,
            seq_id,
            prompt_length,
            cancelled: false,
        })
    }

    pub fn get_num_cached_tokens(&self) -> usize {
        let engine = self.engine.read();
        engine.get_num_cached_tokens()
    }

    pub fn get_available_kv_tokens(&self) -> usize {
        let engine = self.engine.read();
        engine.get_available_kv_tokens()
    }
}

pub struct EngineStream {
    engine: Arc<RwLock<LLMEngine>>,
    rx: mpsc::Receiver<StreamItem>,
    finished: bool,
    pub seq_id: usize,
    pub prompt_length: usize,
    cancelled: bool,
}

impl EngineStream {
    pub fn cancel(&mut self) {
        self.cancelled = true;
        let mut engine_guard = self.engine.write();
        engine_guard.cancel(self.seq_id);
    }

    pub async fn recv(&mut self) -> Option<StreamItem> {
        if self.finished {
            return None;
        }
        let item = self.rx.recv().await;
        if matches!(item, Some(StreamItem::Done(_) | StreamItem::Error(_))) {
            self.finished = true;
        }
        item
    }

    pub fn recv_blocking(&mut self) -> Option<StreamItem> {
        if self.finished {
            return None;
        }
        let item = GLOBAL_RT.block_on(self.rx.recv());
        if matches!(item, Some(StreamItem::Done(_) | StreamItem::Error(_))) {
            self.finished = true;
        }
        item
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled
    }
}

fn parse_device_ids(device_ids: &str) -> Result<Vec<usize>> {
    device_ids
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            usize::from_str(s.trim())
                .map_err(|e| candle_core::Error::msg(format!("Invalid device id '{s}': {e}")))
        })
        .collect()
}

fn dtype_to_str(dtype: DType) -> String {
    match dtype {
        DType::F16 => "f16".to_string(),
        DType::BF16 => "bf16".to_string(),
        DType::F32 => "f32".to_string(),
        _ => "bf16".to_string(),
    }
}
