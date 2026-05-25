use crate::core::engine::LLMEngine;
use crate::core::engine::StreamItem;
use crate::core::engine::GLOBAL_RT;
use crate::core::GenerationOutput;
use crate::server::run_server;
use crate::transfer::{PdConfig, PdMethod, PdRole};
use crate::utils::chat_template::Message;
use crate::utils::config::{EngineConfig, GenerationConfig, SamplingParams};
use crate::utils::get_dtype;
use crate::utils::reasoning::ReasoningEffort;
use llguidance::api::TopLevelGrammar;
use parking_lot::RwLock;
use pyo3::exceptions::PyStopIteration;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;
use tokio::sync::mpsc;

/// Python wrapper
#[pyclass]
pub struct Engine {
    engine: Arc<RwLock<LLMEngine>>,
    #[pyo3(get, set)]
    econfig: EngineConfig,
}

#[pymethods]
impl Engine {
    #[new]
    #[pyo3(text_signature = "(econfig, dtype)")]
    pub fn new(econfig: EngineConfig, dtype: String) -> PyResult<Self> {
        let dtype_parsed = match dtype.as_str() {
            "f16" | "bf16" | "f32" => get_dtype(Some(dtype)),
            _ => {
                return Err(PyValueError::new_err(
                    "Invalid data type (only f16, bf16 and f32 are supported)",
                ))
            }
        };

        match LLMEngine::new(&econfig, dtype_parsed) {
            Ok(engine) => Ok(Self { engine, econfig }),
            Err(e) => Err(PyValueError::new_err(format!("Engine init failed: {e:?}"))),
        }
    }

    #[pyo3(
        name = "start_server",
        text_signature = "($self, port, with_ui_server)"
    )]
    pub fn start_server(&self, port: usize, with_ui_server: bool) -> PyResult<()> {
        GLOBAL_RT.block_on(async move {
            run_server(
                self.engine.clone(),
                self.econfig.clone(),
                port,
                with_ui_server,
            )
            .await
            .map_err(|e| PyValueError::new_err(format!("Server error: {e:?}")))?;
            Ok::<(), PyErr>(())
        })?;

        Ok(())
    }

    #[pyo3(
        name = "generate_sync",
        text_signature = "($self, params, message_list)"
    )]
    pub fn generate_sync(
        &mut self,
        params: Vec<SamplingParams>,
        message_list: Vec<Vec<Message>>,
    ) -> PyResult<Vec<GenerationOutput>> {
        tokio::task::block_in_place(|| {
            GLOBAL_RT.block_on(async {
                let (receivers, tokenizer) = {
                    let mut engine = self.engine.write();
                    (
                        engine
                            .generate_sync(&params, &message_list, None, &Vec::new(), &None)
                            .map_err(|e| {
                                PyValueError::new_err(format!("generate_sync failed: {:?}", e))
                            })?,
                        Arc::new(engine.tokenizer.clone()),
                    )
                };

                let results = LLMEngine::collect_sync_results(receivers, tokenizer, None)
                    .await
                    .map_err(|e| {
                        PyValueError::new_err(format!("collect_sync_results failed: {:?}", e))
                    })?;

                // GenerationOutput is returned directly
                let outputs: Vec<GenerationOutput> = results;

                Ok(outputs)
            })
        })
    }

    #[pyo3(name = "generate_stream", text_signature = "($self, params, messages)")]
    pub fn generate_stream(
        &mut self,
        params: SamplingParams,
        messages: Vec<Message>,
    ) -> PyResult<(usize, usize, EngineStream)> {
        let (seq_id, prompt_length, _prefilled_reasoning_end, stream) = {
            let mut engine = self.engine.write();
            engine
                .generate_stream(&params, &messages, None, &Vec::new(), &None)
                .map_err(|e| PyValueError::new_err(format!("stream error: {:?}", e)))?
        };

        Ok((
            seq_id,
            prompt_length,
            EngineStream {
                engine: self.engine.clone(),
                finished: false,
                seq_id,
                prompt_length,
                cancelled: false,
                rx: std::sync::Mutex::new(stream),
            },
        ))
    }

    #[pyo3(name = "get_num_cached_tokens", text_signature = "($self)")]
    pub fn get_num_cached_tokens(&mut self) -> PyResult<usize> {
        let engine = self.engine.read();
        Ok(engine.get_num_cached_tokens())
    }

    /// Per-sequence prefix-cache hit count. Returns `None` when the seq id
    /// is unknown (e.g. swept long ago and dropped from the side-cache).
    #[pyo3(
        name = "get_num_cached_tokens_for_seq",
        text_signature = "($self, seq_id)"
    )]
    pub fn get_num_cached_tokens_for_seq(&mut self, seq_id: usize) -> PyResult<Option<usize>> {
        let engine = self.engine.read();
        Ok(engine.get_num_cached_tokens_for_seq(seq_id))
    }

    #[pyo3(name = "get_available_kv_tokens", text_signature = "($self)")]
    pub fn get_available_kv_tokens(&mut self) -> PyResult<usize> {
        let engine = self.engine.read();
        Ok(engine.get_available_kv_tokens())
    }
}

#[pyclass(name = "StreamItem")]
#[derive(Clone)]
pub struct PyStreamItem(StreamItem);
use pyo3::IntoPyObjectExt;
#[pymethods]
impl PyStreamItem {
    /// A string representing the type of the stream item.
    /// e.g., "TOKEN", "DONE", "ERROR".
    #[getter]
    fn datatype(&self) -> &'static str {
        match self.0 {
            StreamItem::Token(_, _) => "TOKEN",
            StreamItem::TokenID(_) => "TOKEN_ID",
            StreamItem::Completion(_) => "COMPLETION",
            StreamItem::Done(_) => "DONE",
            StreamItem::Error(_) => "ERROR",
        }
    }

    /// The data associated with the stream item. The Python type of this
    /// data depends on the `type`.
    /// - "TOKEN": str
    /// - "DONE": tuple[int, int, int, int]
    /// - "ERROR": str
    /// etc.
    #[getter]
    fn data(&self, py: Python) -> PyResult<Py<PyAny>> {
        match &self.0 {
            StreamItem::Token(s, id) => (s, id).into_py_any(py),
            StreamItem::TokenID(id) => id.into_py_any(py),
            StreamItem::Completion(c) => (c.0, c.1, c.2, c.3.clone()).into_py_any(py),
            StreamItem::Done(d) => (d.0, d.1, d.2, d.3).into_py_any(py),
            StreamItem::Error(e) => e.into_py_any(py),
        }
    }

    fn __repr__(&self) -> String {
        format!("<StreamItem type={}>", self.datatype())
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

#[pyclass]
#[allow(unused_variables)]
pub struct EngineStream {
    engine: Arc<RwLock<LLMEngine>>,
    rx: std::sync::Mutex<mpsc::Receiver<StreamItem>>,
    #[pyo3(get, set)]
    finished: bool,
    #[pyo3(get, set)]
    seq_id: usize,
    #[pyo3(get, set)]
    prompt_length: usize,
    #[pyo3(get, set)]
    cancelled: bool, // User cancellation flag
}

#[pymethods]
impl EngineStream {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn cancel(mut slf: PyRefMut<Self>) {
        slf.cancelled = true;
        let mut engine_guard = slf.engine.write();
        engine_guard.cancel(slf.seq_id);
    }

    fn __next__(&mut self) -> PyResult<PyStreamItem> {
        // If the stream was already marked as finished on the previous
        // iteration, stop now.
        if self.finished {
            return Err(PyStopIteration::new_err(""));
        }

        let mut rx = self.rx.lock().unwrap();

        // Block and wait for the next item from the channel.
        match GLOBAL_RT.block_on(rx.recv()) {
            Some(item) => {
                // If this is a terminal item (Done or Error), we'll return it
                // to the user this time, but set a flag so that the *next*
                // call to __next__ raises StopIteration.
                if matches!(item, StreamItem::Done(_) | StreamItem::Error(_)) {
                    self.finished = true;
                }

                // Wrap the Rust enum in our PyO3 class and return it.
                Ok(PyStreamItem(item))
            }
            // The channel is empty and disconnected, so the stream is finished.
            _ => {
                self.finished = true;
                Err(PyStopIteration::new_err("[DONE]"))
            }
        }
    }
}

#[pymethods]
impl Message {
    #[new]
    #[pyo3(signature = (role, content, num_images=0))]
    pub fn new(role: String, content: String, num_images: usize) -> Self {
        Message {
            role,
            content,
            num_images,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }
}

#[pymethods]
impl EngineConfig {
    #[new]
    #[pyo3(signature = (model_id=None, weight_path=None, weight_file=None,
        hf_token=None, hf_token_path=None, enforce_parser=None,
        max_num_seqs=Some(32), config_model_len=None, max_model_len=Some(1024), max_tokens=None,
        isq=None, num_shards=Some(1), device_ids=None,
        generation_cfg=None, seed=None, disable_prefix_cache=false, prefix_cache_max_tokens=None,
        kvcache_dtype=None, server_mode=None, cpu_mem_fold=None, kv_fraction=None, mamba_fraction=None, pd_config=None,
        mcp_command=None, mcp_config=None, mcp_args=None,
        tool_prompt_template=None,
        pd_server_prefix_cache_ratio=None, pd_client_prefix_cache_ratio=None, yarn_scaling_factor=None,
        disable_reasoning=false, disable_cuda_graph=false, prefill_chunk_size=Some(8192),))]
    pub fn new(
        model_id: Option<String>,
        weight_path: Option<String>,
        weight_file: Option<String>,
        hf_token: Option<String>,
        hf_token_path: Option<String>,
        enforce_parser: Option<String>,
        max_num_seqs: Option<usize>,
        config_model_len: Option<usize>,
        max_model_len: Option<usize>,
        max_tokens: Option<usize>,
        isq: Option<String>,
        num_shards: Option<usize>,
        device_ids: Option<Vec<usize>>,
        generation_cfg: Option<GenerationConfig>,
        seed: Option<u64>,
        disable_prefix_cache: bool,
        prefix_cache_max_tokens: Option<usize>,
        kvcache_dtype: Option<String>,
        server_mode: Option<bool>,
        cpu_mem_fold: Option<f32>,
        kv_fraction: Option<f32>,
        mamba_fraction: Option<f32>,
        pd_config: Option<PdConfig>,
        mcp_command: Option<String>,
        mcp_config: Option<String>,
        mcp_args: Option<Vec<String>>,
        tool_prompt_template: Option<String>,
        pd_server_prefix_cache_ratio: Option<f32>,
        pd_client_prefix_cache_ratio: Option<f32>,
        yarn_scaling_factor: Option<f64>,
        disable_reasoning: bool,
        disable_cuda_graph: bool,
        prefill_chunk_size: Option<usize>,
    ) -> Self {
        let mut device_ids = device_ids.unwrap_or_default();
        if device_ids.is_empty() {
            device_ids.push(0);
        }

        Self {
            model_id,
            weight_path,
            weight_file,
            hf_token,
            hf_token_path,
            enforce_parser,
            num_blocks: 128, //placeholder
            kv_fraction,
            mamba_fraction,
            cpu_mem_fold,
            kvcache_memory_bytes: 0, //placeholder
            mamba_memory_bytes: 0,
            mamba_slot_bytes: 0,
            mamba_cache_capacity: None,
            block_size: if cfg!(feature = "metal") { 32 } else { 64 },
            max_num_seqs: max_num_seqs.unwrap_or(32),
            max_num_batched_tokens: 32768, //placeholder
            config_model_len,
            max_model_len, //placeholder
            max_tokens,
            isq,
            num_shards,
            device_ids: Some(device_ids),
            generation_cfg,
            seed,
            prefix_cache: Some(!disable_prefix_cache),
            prefix_cache_max_tokens,
            kvcache_dtype: if let Some(ref s) = kvcache_dtype {
                crate::utils::config::KvCacheDtype::from_str_opt(s)
                    .unwrap_or(crate::utils::config::KvCacheDtype::Auto)
            } else {
                crate::utils::config::KvCacheDtype::Auto
            },
            server_mode,
            pd_config,
            mcp_command,
            mcp_config,
            mcp_args,
            tool_prompt_template,
            pd_server_prefix_cache_ratio,
            pd_client_prefix_cache_ratio,
            yarn_scaling_factor,
            disable_reasoning,
            disable_cuda_graph,
            prefill_chunk_size: crate::utils::config::normalize_prefill_chunk_size(
                prefill_chunk_size.unwrap_or(crate::utils::config::DEFAULT_PREFILL_CHUNK_SIZE),
            ),
        }
    }
}

#[pymethods]
impl SamplingParams {
    #[new]
    #[pyo3(signature = (temperature=None, max_tokens=None,
        ignore_eos=Some(false), top_k=None, top_p=None, session_id=None,
        frequency_penalty=None, presence_penalty=None, thinking=None,
        grammar_json=None))]
    pub fn new(
        temperature: Option<f32>,
        max_tokens: Option<usize>,
        ignore_eos: Option<bool>,
        top_k: Option<isize>,
        top_p: Option<f32>,
        session_id: Option<String>,
        frequency_penalty: Option<f32>,
        presence_penalty: Option<f32>,
        thinking: Option<bool>,
        grammar_json: Option<String>,
    ) -> Self {
        // Convert grammar_json to TopLevelGrammar if present
        let grammar = grammar_json
            .as_ref()
            .and_then(|s| serde_json::from_str::<TopLevelGrammar>(s).ok());

        Self {
            temperature,
            max_tokens,
            ignore_eos: ignore_eos.unwrap_or(false),
            top_k,
            top_p,
            session_id,
            frequency_penalty,
            presence_penalty,
            mcp_mode: None,
            stop_sequences: None,
            stop_token_ids: None,
            thinking,
            grammar_json,
            grammar,
            reasoning_effort: None,
        }
    }

    #[staticmethod]
    pub fn new_with_max_tokens(max_tokens: usize) -> Self {
        Self {
            temperature: None,
            max_tokens: Some(max_tokens),
            ignore_eos: false,
            top_k: None,
            top_p: None,
            session_id: None,
            frequency_penalty: None,
            presence_penalty: None,
            mcp_mode: None,
            stop_sequences: None,
            stop_token_ids: None,
            thinking: None,
            grammar_json: None,
            grammar: None,
            reasoning_effort: None,
        }
    }

    #[getter]
    fn grammar_json(&self) -> Option<String> {
        self.grammar
            .as_ref()
            .and_then(|g| serde_json::to_string(g).ok())
    }

    #[setter]
    fn set_grammar_json(&mut self, value: Option<String>) {
        self.grammar_json = value.clone();
        // Also update grammar from JSON if provided
        if let Some(ref s) = value {
            self.grammar = serde_json::from_str::<TopLevelGrammar>(s).ok();
        } else {
            self.grammar = None;
        }
    }

    #[getter]
    fn reasoning_effort(&self) -> Option<String> {
        self.reasoning_effort.as_ref().map(|effort| match effort {
            ReasoningEffort::None => "none".to_string(),
            ReasoningEffort::Low => "low".to_string(),
            ReasoningEffort::Medium => "medium".to_string(),
            ReasoningEffort::High => "high".to_string(),
            ReasoningEffort::ChainOfThought => "chain_of_thought".to_string(),
        })
    }

    #[setter]
    fn set_reasoning_effort(&mut self, value: Option<String>) {
        self.reasoning_effort = value.map(ReasoningEffort::from_str);
    }
}

#[pymethods]
impl GenerationConfig {
    #[new]
    #[pyo3(signature = (temperature=None, top_p=None, top_k=None, frequency_penalty=None, presence_penalty=None))]
    pub fn new(
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<isize>,
        frequency_penalty: Option<f32>,
        presence_penalty: Option<f32>,
    ) -> Self {
        Self {
            temperature,
            top_p,
            top_k,
            frequency_penalty,
            presence_penalty,
            bos_token_id: None,
            eos_token_id: None,
        }
    }
}

#[pymethods]
impl PdConfig {
    #[new]
    #[pyo3(signature = (role, method, url=None))]
    pub fn new(role: PdRole, method: PdMethod, url: Option<String>) -> Self {
        #[cfg(not(feature = "cuda"))]
        if url.is_none() {
            panic!("Non-CUDA platform does not support LocalIPC, please provide pd-url (e.g., server: 0.0.0.0:8100, client: server_id:8100)!");
        }
        Self { role, method, url }
    }
}
