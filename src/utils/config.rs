// src/utils/config.rs
use crate::transfer::PdConfig;
use crate::utils::reasoning::ReasoningEffort;
use llguidance::api::TopLevelGrammar;
#[cfg(feature = "python")]
use pyo3::pyclass;
use serde::de::value::SeqAccessDeserializer;
use serde::de::{Deserializer, Visitor};
use serde::ser::Error as _;
use serde::{Deserialize, Serialize, Serializer};
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvCacheDtype {
    Auto,
    Fp8,
    Turbo8,
    Turbo4,
    Turbo3,
}

impl KvCacheDtype {
    pub fn is_turboquant(&self) -> bool {
        matches!(self, Self::Turbo8 | Self::Turbo4 | Self::Turbo3)
    }

    pub fn is_fp8_keys(&self) -> bool {
        matches!(self, Self::Fp8 | Self::Turbo8)
    }

    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "auto" | "bf16" | "bfloat16" => Some(Self::Auto),
            "fp8" | "e4m3" => Some(Self::Fp8),
            "turbo8" | "k8v4" => Some(Self::Turbo8),
            "turbo4" | "4bit" => Some(Self::Turbo4),
            "turbo3" | "k3v4" => Some(Self::Turbo3),
            _ => None,
        }
    }
}

impl Default for KvCacheDtype {
    fn default() -> Self {
        Self::Auto
    }
}

impl std::fmt::Display for KvCacheDtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::Fp8 => write!(f, "fp8"),
            Self::Turbo8 => write!(f, "turbo8"),
            Self::Turbo4 => write!(f, "turbo4"),
            Self::Turbo3 => write!(f, "turbo3"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum EosTokenId {
    Single(u32),
    Multiple(Vec<u32>),
}

impl<'de> Deserialize<'de> for EosTokenId {
    fn deserialize<D>(deserializer: D) -> Result<EosTokenId, D::Error>
    where
        D: Deserializer<'de>,
    {
        if deserializer.is_human_readable() {
            // For JSON: deserialize as "untagged" using a visitor
            struct EosTokenIdVisitor;

            impl<'de> Visitor<'de> for EosTokenIdVisitor {
                type Value = EosTokenId;

                fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("a u32 or a sequence of u32s")
                }

                // Handle a single number
                fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E> {
                    Ok(EosTokenId::Single(v as u32))
                }

                // Handle an array of numbers
                fn visit_seq<A>(self, seq: A) -> Result<Self::Value, A::Error>
                where
                    A: serde::de::SeqAccess<'de>,
                {
                    let vals = Vec::<u32>::deserialize(SeqAccessDeserializer::new(seq))?;
                    Ok(EosTokenId::Multiple(vals))
                }
            }

            deserializer.deserialize_any(EosTokenIdVisitor)
        } else {
            // For Bincode: deserialize as "tagged"
            let bincode_id = BincodeEosTokenId::deserialize(deserializer)?;
            let id = match bincode_id {
                BincodeEosTokenId::Single(v) => EosTokenId::Single(v),
                BincodeEosTokenId::Multiple(v) => EosTokenId::Multiple(v),
            };
            Ok(id)
        }
    }
}

impl Serialize for EosTokenId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if serializer.is_human_readable() {
            // For JSON: serialize as "untagged"
            match self {
                EosTokenId::Single(v) => v.serialize(serializer),
                EosTokenId::Multiple(v) => v.serialize(serializer),
            }
        } else {
            // For Bincode: serialize as "tagged"
            let bincode_id = match self {
                EosTokenId::Single(v) => BincodeEosTokenId::Single(*v),
                EosTokenId::Multiple(v) => BincodeEosTokenId::Multiple(v.clone()),
            };
            bincode_id.serialize(serializer)
        }
    }
}

impl EosTokenId {
    /// Merge `other` into `self`, returning the combined token set.
    /// - Single + Single => Multiple([a, b])
    /// - Single + Multiple => Multiple([a, ...])
    /// - Multiple + Single => Multiple([... , b])
    /// - Multiple + Multiple => Multiple([... , ...])
    pub fn merge(self, other: EosTokenId) -> EosTokenId {
        let mut out = self.into_vec();
        out.extend(other.into_vec());
        EosTokenId::Multiple(out)
    }

    /// Like merge, but de-duplicates while preserving first-seen order.
    pub fn merge_dedup(self, other: EosTokenId) -> EosTokenId {
        use std::collections::HashSet;

        let mut seen = HashSet::<u32>::new();
        let mut out = Vec::<u32>::new();

        for id in self.into_vec().into_iter().chain(other.into_vec()) {
            if seen.insert(id) {
                out.push(id);
            }
        }
        EosTokenId::Multiple(out)
    }

    pub fn to_vec(&self) -> Vec<u32> {
        match self {
            EosTokenId::Single(x) => vec![*x],
            EosTokenId::Multiple(v) => v.clone(),
        }
    }

    fn into_vec(self) -> Vec<u32> {
        match self {
            EosTokenId::Single(x) => vec![x],
            EosTokenId::Multiple(v) => v,
        }
    }
}

fn serialize_optional_grammar<S>(
    grammar: &Option<TopLevelGrammar>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    if serializer.is_human_readable() {
        grammar.serialize(serializer)
    } else {
        let encoded = grammar
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .map_err(S::Error::custom)?;
        encoded.serialize(serializer)
    }
}

fn deserialize_optional_grammar<'de, D>(
    deserializer: D,
) -> Result<Option<TopLevelGrammar>, D::Error>
where
    D: Deserializer<'de>,
{
    if deserializer.is_human_readable() {
        Option::<TopLevelGrammar>::deserialize(deserializer)
    } else {
        let encoded = Option::<String>::deserialize(deserializer)?;
        encoded
            .map(|json| serde_json::from_str(&json).map_err(serde::de::Error::custom))
            .transpose()
    }
}

// To make the "tagged" logic work for bincode, we need a separate
// definition of the enum with derived traits. We keep it private inside this module.
#[derive(Serialize, Deserialize)]
enum BincodeEosTokenId {
    Single(u32),
    Multiple(Vec<u32>),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MoEConfig {
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: Option<usize>,
    #[serde(alias = "n_routed_experts", alias = "num_local_experts")]
    pub num_experts: Option<usize>,
    pub mlp_only_layers: Option<Vec<usize>>,
    pub decoder_sparse_step: Option<usize>,
    #[serde(default)]
    pub norm_topk_prob: bool,
    pub num_experts_per_tok: usize,
    pub routed_scaling_factor: Option<f64>,
    pub first_k_dense_replace: Option<usize>,
    pub n_shared_experts: Option<usize>,
    pub n_group: Option<usize>,
    pub topk_group: Option<usize>,
    #[serde(default)]
    pub scoring_func: Option<String>,
    #[serde(default)]
    pub topk_method: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RopeScalingValue {
    Bool(bool),
    Number(f64),
    NumberArray(Vec<f64>),
    String(String),
}

impl RopeScalingValue {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            RopeScalingValue::Number(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            RopeScalingValue::String(v) => Some(v),
            _ => None,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Config {
    pub architectures: Option<Vec<String>>,
    pub head_dim: Option<usize>,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub max_model_len: Option<usize>,
    #[serde(default, alias = "ffn_hidden_size", alias = "feed_forward_length")]
    pub intermediate_size: usize,
    pub rms_norm_eps: f64,
    pub vocab_size: Option<usize>,
    pub rope_theta: Option<f64>,
    pub attention_bias: Option<bool>,
    pub qkv_bias: Option<bool>,
    pub attn_output_gate: Option<bool>,
    pub attn_logit_softcapping: Option<f64>,
    pub final_logit_softcapping: Option<f64>,
    pub tie_word_embeddings: Option<bool>,
    pub bos_token_id: Option<usize>,
    pub eos_token_id: Option<EosTokenId>,
    pub use_sliding_window: Option<bool>,
    pub sliding_window: Option<usize>,
    pub max_window_layers: Option<usize>,
    pub partial_rotary_factor: Option<f32>,
    #[serde(alias = "hidden_activation")]
    pub hidden_act: candle_nn::Activation,
    #[serde(alias = "rope_parameters")]
    pub rope_scaling: Option<HashMap<String, RopeScalingValue>>,
    pub quant: Option<String>,
    pub moe_cfg: Option<MoEConfig>,
    #[serde(default)]
    pub kvcache_dtype: KvCacheDtype,
    pub quantization_config: Option<QuantConfig>,
    pub is_multi_model: Option<bool>,
    pub extra_config_json: Option<String>,
    #[serde(default)]
    pub is_f16_mode: bool,
}

impl Config {
    pub fn apply_generation_cfg(&mut self, generation_cfg: Option<&GenerationConfig>) {
        let Some(gcfg) = generation_cfg else { return };

        // BOS merge (fill if missing; config wins)
        if self.bos_token_id.is_none() {
            self.bos_token_id = gcfg.bos_token_id;
        }

        // EOS merge (combine if both present)
        self.eos_token_id = match (self.eos_token_id.take(), gcfg.eos_token_id.as_ref()) {
            (None, None) => None,
            (None, Some(e)) => Some(e.clone()),
            (Some(e), None) => Some(e),
            (Some(e), Some(other)) => Some(e.merge(other.clone())),
        };
    }

    pub fn higher_precision_required(&self) -> bool {
        self.is_f16_mode
            || self.quant.is_some()
            || self
                .quantization_config
                .as_ref()
                .is_some_and(|cfg| matches!(cfg.quant_method.as_str(), "mxfp4" | "nvfp4"))
    }
}

pub const DEFAULT_PREFILL_CHUNK_SIZE: usize = 8 * 1024;
pub const MIN_PREFILL_CHUNK_SIZE: usize = 1024;
pub const MAX_PREFILL_CHUNK_SIZE: usize = 32 * 1024;

pub fn default_prefill_chunk_size() -> usize {
    DEFAULT_PREFILL_CHUNK_SIZE
}

pub fn normalize_prefill_chunk_size(size: usize) -> usize {
    let rounded = (size.saturating_add(MIN_PREFILL_CHUNK_SIZE / 2) / MIN_PREFILL_CHUNK_SIZE)
        * MIN_PREFILL_CHUNK_SIZE;
    rounded.clamp(MIN_PREFILL_CHUNK_SIZE, MAX_PREFILL_CHUNK_SIZE)
}

#[cfg(not(feature = "python"))]
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EngineConfig {
    pub model_id: Option<String>,
    pub weight_path: Option<String>,
    pub weight_file: Option<String>,
    pub enforce_parser: Option<String>,
    pub hf_token: Option<String>,
    pub hf_token_path: Option<String>,
    pub num_blocks: usize,
    pub kv_fraction: Option<f32>, // After loading the model, the remaining percent of gpu used for kvcache
    pub mamba_fraction: Option<f32>, // Percent of cache budget reserved for hybrid mamba states
    pub cpu_mem_fold: Option<f32>, // the percentage of gpu kvcache: 0.1x to 10x, default 1.0x
    pub kvcache_memory_bytes: usize,
    #[serde(default)]
    pub mamba_memory_bytes: usize,
    #[serde(default)]
    pub mamba_slot_bytes: usize,
    #[serde(default)]
    pub mamba_cache_capacity: Option<usize>,
    pub block_size: usize,
    pub max_num_seqs: usize,
    pub max_num_batched_tokens: usize,
    pub config_model_len: Option<usize>,
    pub max_model_len: Option<usize>,
    pub max_tokens: Option<usize>,
    pub isq: Option<String>,
    pub num_shards: Option<usize>,
    pub device_ids: Option<Vec<usize>>,
    pub generation_cfg: Option<GenerationConfig>,
    pub seed: Option<u64>,
    pub prefix_cache: Option<bool>,
    pub prefix_cache_max_tokens: Option<usize>,
    #[serde(default)]
    pub kvcache_dtype: KvCacheDtype,
    pub server_mode: Option<bool>,
    pub pd_config: Option<PdConfig>,
    pub mcp_command: Option<String>,
    pub mcp_config: Option<String>,
    pub mcp_args: Option<Vec<String>>,
    pub tool_prompt_template: Option<String>,
    pub pd_server_prefix_cache_ratio: Option<f32>,
    pub pd_client_prefix_cache_ratio: Option<f32>,
    pub yarn_scaling_factor: Option<f64>,
    #[serde(default)]
    pub disable_reasoning: bool,
    #[serde(default)]
    pub disable_cuda_graph: bool,
    #[serde(default = "default_prefill_chunk_size")]
    pub prefill_chunk_size: usize,
}

#[cfg(feature = "python")]
#[pyclass]
#[allow(unused_variables)]
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EngineConfig {
    #[pyo3(get, set)]
    pub model_id: Option<String>,
    #[pyo3(get, set)]
    pub weight_path: Option<String>,
    #[pyo3(get, set)]
    pub weight_file: Option<String>,
    #[pyo3(get, set)]
    pub enforce_parser: Option<String>,
    #[pyo3(get, set)]
    pub hf_token: Option<String>,
    #[pyo3(get, set)]
    pub hf_token_path: Option<String>,
    #[pyo3(get, set)]
    pub num_blocks: usize,
    #[pyo3(get, set)]
    pub cpu_mem_fold: Option<f32>,
    #[pyo3(get, set)]
    pub kv_fraction: Option<f32>,
    #[pyo3(get, set)]
    pub mamba_fraction: Option<f32>,
    pub block_size: usize,
    pub kvcache_memory_bytes: usize,
    #[serde(default)]
    pub mamba_memory_bytes: usize,
    #[serde(default)]
    pub mamba_slot_bytes: usize,
    #[pyo3(get, set)]
    pub mamba_cache_capacity: Option<usize>,
    #[pyo3(get, set)]
    pub max_num_seqs: usize,
    #[pyo3(get, set)]
    pub max_num_batched_tokens: usize,
    #[pyo3(get, set)]
    pub max_model_len: Option<usize>,
    #[pyo3(get, set)]
    pub config_model_len: Option<usize>,
    #[pyo3(get, set)]
    pub max_tokens: Option<usize>,
    #[pyo3(get, set)]
    pub isq: Option<String>,
    #[pyo3(get, set)]
    pub num_shards: Option<usize>,
    #[pyo3(get, set)]
    pub device_ids: Option<Vec<usize>>,
    #[pyo3(get, set)]
    pub generation_cfg: Option<GenerationConfig>,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
    #[pyo3(get, set)]
    pub prefix_cache: Option<bool>,
    #[pyo3(get, set)]
    pub prefix_cache_max_tokens: Option<usize>,
    #[serde(default)]
    pub kvcache_dtype: KvCacheDtype,
    #[pyo3(get, set)]
    pub server_mode: Option<bool>,
    #[pyo3(get, set)]
    pub pd_config: Option<PdConfig>,
    #[pyo3(get, set)]
    pub mcp_command: Option<String>,
    #[pyo3(get, set)]
    pub mcp_config: Option<String>,
    #[pyo3(get, set)]
    pub mcp_args: Option<Vec<String>>,
    #[pyo3(get, set)]
    pub tool_prompt_template: Option<String>,
    #[pyo3(get, set)]
    pub pd_server_prefix_cache_ratio: Option<f32>,
    #[pyo3(get, set)]
    pub pd_client_prefix_cache_ratio: Option<f32>,
    #[pyo3(get, set)]
    pub yarn_scaling_factor: Option<f64>,
    #[pyo3(get, set)]
    pub disable_reasoning: bool,
    #[pyo3(get, set)]
    pub disable_cuda_graph: bool,
    #[pyo3(get, set)]
    #[serde(default = "default_prefill_chunk_size")]
    pub prefill_chunk_size: usize,
}

impl EngineConfig {
    pub fn effective_prefill_chunk_size(&self) -> usize {
        #[cfg(feature = "metal")]
        {
            normalize_prefill_chunk_size(self.prefill_chunk_size / 2)
        }
        #[cfg(not(feature = "metal"))]
        {
            normalize_prefill_chunk_size(self.prefill_chunk_size)
        }
    }
}

#[cfg(not(feature = "python"))]
impl EngineConfig {
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
            cpu_mem_fold,
            kv_fraction,
            mamba_fraction,
            kvcache_memory_bytes: 0, //placeholder
            mamba_memory_bytes: 0,
            mamba_slot_bytes: 0,
            mamba_cache_capacity: None,
            block_size: if cfg!(feature = "metal") { 32 } else { 64 },
            max_num_seqs: max_num_seqs.unwrap_or(32),
            max_num_batched_tokens: max_num_seqs.unwrap_or(32) * 1024, //placeholder
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
                KvCacheDtype::from_str_opt(s).unwrap_or(KvCacheDtype::Auto)
            } else {
                KvCacheDtype::Auto
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
            prefill_chunk_size: normalize_prefill_chunk_size(
                prefill_chunk_size.unwrap_or(DEFAULT_PREFILL_CHUNK_SIZE),
            ),
        }
    }
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct TokenizerConfig {
    pub model_max_length: Option<f64>,
    pub add_bos_token: Option<bool>,
    pub add_eos_token: Option<bool>,
    pub chat_template: Option<String>,
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
}

#[cfg(not(feature = "python"))]
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SamplingParams {
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
    pub ignore_eos: bool,
    pub top_k: Option<isize>,
    pub top_p: Option<f32>,
    pub session_id: Option<String>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip)]
    pub stop_token_ids: Option<Vec<Vec<u32>>>,
    #[serde(alias = "enable_thinking")]
    pub thinking: Option<bool>, // enable reasoning
    /// Tool mode for tool call handling.
    /// If Some(true), external tools are enabled and stream finishes at </tool_call>.
    #[serde(default)]
    pub mcp_mode: Option<bool>,
    /// Grammar constraint as TopLevelGrammar for RPC serialization
    #[serde(default)]
    #[serde(
        serialize_with = "serialize_optional_grammar",
        deserialize_with = "deserialize_optional_grammar"
    )]
    pub grammar: Option<TopLevelGrammar>,
    #[serde(default)]
    pub grammar_json: Option<String>,
    /// Reasoning effort level for OpenAI-compatible reasoning API
    #[serde(default)]
    pub reasoning_effort: Option<ReasoningEffort>,
}

#[cfg(feature = "python")]
#[pyclass]
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SamplingParams {
    #[pyo3(get, set)]
    pub temperature: Option<f32>,
    #[pyo3(get, set)]
    pub max_tokens: Option<usize>,
    #[pyo3(get, set)]
    pub ignore_eos: bool,
    #[pyo3(get, set)]
    pub top_k: Option<isize>,
    #[pyo3(get, set)]
    pub top_p: Option<f32>,
    #[pyo3(get, set)]
    pub session_id: Option<String>,
    #[pyo3(get, set)]
    pub frequency_penalty: Option<f32>,
    #[pyo3(get, set)]
    pub presence_penalty: Option<f32>,
    #[pyo3(get, set)]
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip)]
    pub stop_token_ids: Option<Vec<Vec<u32>>>,
    /// Tool mode for tool call handling.
    /// If Some(true), external tools are enabled and stream finishes at </tool_call>.
    #[pyo3(get, set)]
    pub mcp_mode: Option<bool>,
    #[pyo3(get, set)]
    #[serde(alias = "enable_thinking")]
    pub thinking: Option<bool>,
    /// Grammar constraint as TopLevelGrammar for RPC serialization
    #[serde(default)]
    #[serde(
        serialize_with = "serialize_optional_grammar",
        deserialize_with = "deserialize_optional_grammar"
    )]
    pub grammar: Option<TopLevelGrammar>,
    /// Grammar constraint as JSON string for Python API
    #[pyo3(get, set)]
    pub grammar_json: Option<String>,
    /// Reasoning effort level for OpenAI-compatible reasoning API
    pub reasoning_effort: Option<ReasoningEffort>,
}

#[cfg(not(feature = "python"))]
impl SamplingParams {
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
        reasoning_effort: Option<ReasoningEffort>,
    ) -> Self {
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
            grammar: None,
            grammar_json: None,
            reasoning_effort,
        }
    }

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
            grammar: None,
            grammar_json: None,
            reasoning_effort: None,
        }
    }
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: None,
            max_tokens: Some(16384),
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
            grammar: None,
            grammar_json: None,
            reasoning_effort: None,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ModelType {
    Qwen3,
    Qwen3MoE,
    Qwen3_5,
    Qwen3_5MoE,
    LLaMa,
    Gemma,
    Gemma3,
    Gemma4,
    Phi,
    Phi4,
    Mistral,
    GLM4,
    GLM4MoE,
    GLM4MoeLite,
    Yi,
    StableLM,
    DeepSeek,
    Mistral3VL,
    Qwen3VL,
    LLaMa4,
    MiniMax,
}

#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Randomness of sampling.
    /// rec. default = 1
    pub temperature: Option<f32>,
    /// Cumulative prob of the top tokens to consider, must be in (0, 1]. Set 1 to consider all toks.  
    /// rec. default = 1    
    pub top_p: Option<f32>,
    /// Control the number of top tokens to consider, set -1 to consider all.
    /// rec. default = -1
    pub top_k: Option<isize>,

    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,

    pub bos_token_id: Option<usize>,
    pub eos_token_id: Option<EosTokenId>,
}

/// Match a module path against an ignore pattern.
/// Supports three pattern types:
///   - `re:` prefix → regex match
///   - Contains `*` → glob match (converted to regex: `*` becomes `.*`)
///   - Otherwise → literal suffix matching
pub fn match_ignore_pattern(module_path: &str, pattern: &str) -> bool {
    if let Some(re_pat) = pattern.strip_prefix("re:") {
        if let Ok(re) = regex::Regex::new(re_pat) {
            return re.is_match(module_path);
        }
        return false;
    }
    if pattern.contains('*') {
        let re_pat = format!("^{}$", regex::escape(pattern).replace(r"\*", ".*"));
        if let Ok(re) = regex::Regex::new(&re_pat) {
            return re.is_match(module_path);
        }
        return false;
    }
    let module_path = module_path.trim_end_matches(".weight");
    let item = pattern.trim_end_matches(".weight");
    module_path == item
        || module_path.ends_with(item)
        || module_path.ends_with(&format!(".{item}"))
        || item.ends_with(module_path)
        || item.ends_with(&format!(".{module_path}"))
}

#[derive(Serialize, Deserialize, PartialEq, Clone)]
pub struct QuantConfig {
    #[serde(default)]
    pub quant_method: String,
    #[serde(default)]
    pub bits: usize,
    #[serde(default)]
    pub group_size: i32,
    pub sym: Option<bool>,
    pub desc_act: Option<bool>,
    pub checkpoint_format: Option<String>,
    pub fmt: Option<String>,
    #[serde(default)]
    pub format: Option<String>,
    pub weight_block_size: Option<Vec<usize>>,
    #[serde(default, alias = "ignore")]
    pub modules_to_not_convert: Vec<String>,
    #[serde(default)]
    pub config_groups: Option<serde_json::Value>,
    #[serde(default)]
    pub quant_algo: Option<String>,
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub quantized_layers: Option<serde_json::Value>,
    /// MLX NVFP4 uses U32-packed weights (8 nibbles per U32) instead of U8
    /// (2 nibbles per byte). Set during normalize_compressed_tensors when
    /// an MLX-style `"mode": "nvfp4"` config is detected.
    #[serde(default)]
    pub is_mlx_nvfp4: bool,
}

impl QuantConfig {
    /// Normalizes a quantization config into a canonical quant_method string.
    ///
    /// Handles the following families:
    ///   1. `modelopt` with `quant_algo` == `NVFP4` / `FP4`
    ///   2. `compressed-tensors` with `format` containing `nvfp4` or `mxfp4`
    ///   3. `compressed-tensors` detected from `config_groups` content
    ///   4. MLX-style `"mode": "nvfp4"` — weights are U32-packed (8 nibbles per
    ///      U32) with FP8 E4M3 scales, no global_scale. Repacked to U8 at load.
    ///
    /// Also extracts group_size / bits from config_groups when present.
    pub fn normalize_compressed_tensors(&mut self) {
        if self.quant_method.is_empty() {
            if let Some(mode) = &self.mode {
                if mode.eq_ignore_ascii_case("nvfp4") {
                    self.quant_method = "nvfp4".to_string();
                    self.is_mlx_nvfp4 = true;
                    if self.group_size == 0 {
                        self.group_size = 16;
                    }
                    if self.bits == 0 {
                        self.bits = 4;
                    }
                    return;
                }
            }
        }
        // modelopt: {"quant_method": "modelopt", "quant_algo": "NVFP4"}
        if self.quant_method == "modelopt" {
            if let Some(algo) = &self.quant_algo {
                if algo.eq_ignore_ascii_case("NVFP4") || algo.eq_ignore_ascii_case("FP4") {
                    self.quant_method = "nvfp4".to_string();
                    self.extract_compressed_tensors_params();
                    if self.group_size == 0 {
                        self.group_size = 16;
                    }
                    if self.bits == 0 {
                        self.bits = 4;
                    }
                    return;
                }
                if algo.eq_ignore_ascii_case("FP8") {
                    self.quant_method = "fp8".to_string();
                    return;
                }
                if algo.eq_ignore_ascii_case("MIXED_PRECISION") {
                    if self.detect_nvfp4_from_config_groups()
                        || self.detect_nvfp4_from_quantized_layers()
                    {
                        self.quant_method = "nvfp4".to_string();
                        self.bits = 4;
                        self.group_size = 16;
                    } else if self.detect_fp8_from_quantized_layers() {
                        self.quant_method = "fp8".to_string();
                    }
                    return;
                }
            }
            if self.detect_nvfp4_from_config_groups() {
                self.quant_method = "nvfp4".to_string();
                self.extract_compressed_tensors_params();
                if self.group_size == 0 {
                    self.group_size = 16;
                }
                if self.bits == 0 {
                    self.bits = 4;
                }
                return;
            }
        }

        if self.quant_method != "compressed-tensors" {
            return;
        }

        // compressed-tensors: check format string for nvfp4 or mxfp4
        let format_str = self.format.as_deref().unwrap_or("");

        let is_nvfp4 = format_str.contains("nvfp4") || self.detect_nvfp4_from_config_groups();

        if is_nvfp4 {
            self.quant_method = "nvfp4".to_string();
            self.extract_compressed_tensors_params();
            if self.group_size == 0 {
                self.group_size = 16;
            }
            if self.bits == 0 {
                self.bits = 4;
            }
            return;
        }

        let is_mxfp4 = format_str.contains("mxfp4") || self.detect_mxfp4_from_config_groups();

        if is_mxfp4 {
            self.quant_method = "mxfp4".to_string();
            self.extract_compressed_tensors_params();
        }
    }

    fn detect_nvfp4_from_config_groups(&self) -> bool {
        let groups = match &self.config_groups {
            Some(v) => v,
            None => return false,
        };
        if let Some(obj) = groups.as_object() {
            for (_key, group) in obj {
                // Check group-level format (e.g. "nvfp4-pack-quantized")
                if let Some(fmt) = group.get("format").and_then(|v| v.as_str()) {
                    if fmt.contains("nvfp4") {
                        return true;
                    }
                }
                if let Some(weights) = group.get("weights") {
                    // Check weights-level format
                    if let Some(fmt) = weights.get("format").and_then(|v| v.as_str()) {
                        if fmt.contains("nvfp4") {
                            return true;
                        }
                    }
                    // Detect by parameters: 4-bit float with group_size=16
                    if let Some(num_bits) = weights.get("num_bits").and_then(|v| v.as_u64()) {
                        if num_bits == 4 {
                            let is_float = weights
                                .get("type")
                                .and_then(|v| v.as_str())
                                .map(|t| t == "float")
                                .unwrap_or(false);
                            let gs = weights
                                .get("group_size")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0);
                            if is_float && gs == 16 {
                                return true;
                            }
                        }
                    }
                }
            }
        }
        false
    }

    fn detect_mxfp4_from_config_groups(&self) -> bool {
        let groups = match &self.config_groups {
            Some(v) => v,
            None => return false,
        };
        if let Some(obj) = groups.as_object() {
            for (_key, group) in obj {
                if let Some(fmt) = group.get("format").and_then(|v| v.as_str()) {
                    if fmt.contains("mxfp4") {
                        return true;
                    }
                }
                if let Some(weights) = group.get("weights") {
                    if let Some(fmt) = weights.get("format").and_then(|v| v.as_str()) {
                        if fmt.contains("mxfp4") {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    fn detect_nvfp4_from_quantized_layers(&self) -> bool {
        let layers = match &self.quantized_layers {
            Some(v) => v,
            None => return false,
        };
        if let Some(obj) = layers.as_object() {
            for (_key, layer_info) in obj {
                if let Some(algo) = layer_info.get("quant_algo").and_then(|v| v.as_str()) {
                    if algo.contains("NVFP4") || algo.contains("nvfp4") {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn detect_fp8_from_quantized_layers(&self) -> bool {
        let layers = match &self.quantized_layers {
            Some(v) => v,
            None => return false,
        };
        if let Some(obj) = layers.as_object() {
            for (_key, layer_info) in obj {
                if let Some(algo) = layer_info.get("quant_algo").and_then(|v| v.as_str()) {
                    if algo == "FP8" || algo == "fp8" {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn extract_compressed_tensors_params(&mut self) {
        let groups = match &self.config_groups {
            Some(v) => v.clone(),
            None => return,
        };
        if let Some(obj) = groups.as_object() {
            for (_key, group) in obj {
                if let Some(weights) = group.get("weights") {
                    if self.group_size == 0 {
                        if let Some(gs) = weights.get("group_size").and_then(|v| v.as_i64()) {
                            self.group_size = gs as i32;
                        }
                    }
                    if self.bits == 0 {
                        if let Some(nb) = weights.get("num_bits").and_then(|v| v.as_u64()) {
                            self.bits = nb as usize;
                        }
                    }
                }
            }
        }
    }

    /// Check if a module path should be skipped for this quantization config.
    /// Supports literal paths and `re:` prefixed regex patterns in
    /// `modules_to_not_convert` / `ignore`.
    pub fn should_skip_module(&self, module_path: &str) -> bool {
        if module_path.is_empty() || self.modules_to_not_convert.is_empty() {
            return false;
        }
        self.modules_to_not_convert
            .iter()
            .any(|item| match_ignore_pattern(module_path, item))
    }
}

impl fmt::Debug for QuantConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QuantConfig")
            .field("quant_method", &self.quant_method)
            .field("bits", &self.bits)
            .field("group_size", &self.group_size)
            .field("sym", &self.sym)
            .field("desc_act", &self.desc_act)
            .field("checkpoint_format", &self.checkpoint_format)
            .field("fmt", &self.fmt)
            .field("format", &self.format)
            .field("weight_block_size", &self.weight_block_size)
            .field("modules_to_not_convert", &self.modules_to_not_convert)
            .field("is_mlx_nvfp4", &self.is_mlx_nvfp4)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_match_ignore_literal_exact() {
        assert!(match_ignore_pattern("lm_head", "lm_head"));
        assert!(match_ignore_pattern(
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.q_proj"
        ));
    }

    #[test]
    fn test_match_ignore_literal_suffix() {
        assert!(match_ignore_pattern(
            "model.language_model.layers.0.linear_attn.out_proj",
            "model.language_model.layers.0.linear_attn.out_proj"
        ));
        assert!(match_ignore_pattern("model.lm_head.weight", "lm_head"));
    }

    #[test]
    fn test_match_ignore_regex() {
        assert!(match_ignore_pattern(
            "model.layers.5.self_attn.q_proj",
            "re:.*self_attn.*"
        ));
        assert!(match_ignore_pattern(
            "model.layers.10.linear_attn.in_proj_qkv",
            "re:.*linear_attn.*"
        ));
        assert!(match_ignore_pattern(
            "model.layers.3.mlp.gate",
            "re:.*.mlp.gate$"
        ));
        assert!(!match_ignore_pattern(
            "model.layers.3.mlp.gate_proj",
            "re:.*.mlp.gate$"
        ));
        assert!(match_ignore_pattern(
            "model.visual.blocks.0.attn.qkv",
            "re:.*visual.*"
        ));
        assert!(match_ignore_pattern("mtp.fc", "re:.*mtp.*"));
        assert!(match_ignore_pattern(
            "model.embed_tokens",
            "re:.*embed_tokens.*"
        ));
    }

    #[test]
    fn test_match_ignore_regex_no_false_positive() {
        assert!(!match_ignore_pattern(
            "model.layers.5.mlp.up_proj",
            "re:.*self_attn.*"
        ));
        assert!(!match_ignore_pattern(
            "model.layers.5.mlp.up_proj",
            "re:.*linear_attn.*"
        ));
        assert!(!match_ignore_pattern(
            "model.layers.5.mlp.up_proj",
            "re:.*lm_head.*"
        ));
    }

    #[test]
    fn test_should_skip_module() {
        let cfg = QuantConfig {
            quant_method: "mxfp4".to_string(),
            bits: 4,
            group_size: 32,
            sym: None,
            desc_act: None,
            checkpoint_format: None,
            fmt: None,
            format: Some("mxfp4-pack-quantized".to_string()),
            weight_block_size: None,
            modules_to_not_convert: vec![
                "re:.*self_attn.*".to_string(),
                "re:.*linear_attn.*".to_string(),
                "re:.*.mlp.gate$".to_string(),
                "re:.*lm_head.*".to_string(),
                "re:.*embed_tokens.*".to_string(),
                "re:.*visual.*".to_string(),
                "re:.*mtp.*".to_string(),
            ],
            config_groups: None,
            quant_algo: None,
            mode: None,
            quantized_layers: None,
            is_mlx_nvfp4: false,
        };
        assert!(cfg.should_skip_module("model.layers.0.self_attn.q_proj"));
        assert!(cfg.should_skip_module("model.layers.5.linear_attn.out_proj"));
        assert!(cfg.should_skip_module("model.layers.3.mlp.gate"));
        assert!(!cfg.should_skip_module("model.layers.3.mlp.gate_proj"));
        assert!(!cfg.should_skip_module("model.layers.3.mlp.up_proj"));
        assert!(!cfg.should_skip_module("model.layers.3.mlp.down_proj"));
        assert!(cfg.should_skip_module("lm_head"));
        assert!(cfg.should_skip_module("model.visual.blocks.0.attn.qkv"));
        assert!(cfg.should_skip_module("mtp.fc"));
    }

    #[test]
    fn test_normalize_compressed_tensors_regex_ignore() {
        let json = r#"{
            "quant_method": "compressed-tensors",
            "format": "mxfp4-pack-quantized",
            "config_groups": {
                "group_0": {
                    "format": "mxfp4-pack-quantized",
                    "weights": {"num_bits": 4, "group_size": 32, "strategy": "group", "symmetric": true}
                }
            },
            "ignore": [
                "re:.*self_attn.*",
                "re:.*linear_attn.*",
                "re:.*.mlp.gate$",
                "re:.*lm_head.*",
                "re:.*embed_tokens.*",
                "re:.*visual.*",
                "re:.*mtp.*"
            ]
        }"#;
        let mut cfg: QuantConfig = serde_json::from_str(json).unwrap();
        cfg.normalize_compressed_tensors();
        assert_eq!(cfg.quant_method, "mxfp4");
        assert_eq!(cfg.group_size, 32);
        assert_eq!(cfg.bits, 4);
        assert_eq!(cfg.modules_to_not_convert.len(), 7);
        assert!(cfg.should_skip_module("model.layers.5.self_attn.q_proj"));
        assert!(!cfg.should_skip_module("model.layers.5.mlp.up_proj"));
    }

    #[test]
    fn test_normalize_compressed_tensors_literal_ignore() {
        let json = r#"{
            "quant_method": "compressed-tensors",
            "format": "mxfp4-pack-quantized",
            "config_groups": {
                "group_0": {
                    "weights": {"num_bits": 4, "group_size": 32}
                }
            },
            "ignore": [
                "model.layers.0.linear_attn.out_proj",
                "model.layers.0.linear_attn.in_proj_qkv",
                "lm_head"
            ]
        }"#;
        let mut cfg: QuantConfig = serde_json::from_str(json).unwrap();
        cfg.normalize_compressed_tensors();
        assert_eq!(cfg.quant_method, "mxfp4");
        assert!(cfg.should_skip_module("model.layers.0.linear_attn.out_proj"));
        assert!(cfg.should_skip_module("lm_head"));
        assert!(!cfg.should_skip_module("model.layers.1.mlp.up_proj"));
    }

    #[test]
    fn test_normalize_format_in_config_groups_only() {
        let json = r#"{
            "quant_method": "compressed-tensors",
            "config_groups": {
                "group_0": {
                    "format": "mxfp4-pack-quantized",
                    "weights": {"num_bits": 4, "group_size": 32, "type": "float", "strategy": "group", "symmetric": true}
                }
            },
            "ignore": ["lm_head"]
        }"#;
        let mut cfg: QuantConfig = serde_json::from_str(json).unwrap();
        cfg.normalize_compressed_tensors();
        assert_eq!(cfg.quant_method, "mxfp4");
        assert_eq!(cfg.group_size, 32);
    }

    #[test]
    fn test_olka_4b_config() {
        let json = r#"{
            "quant_method": "compressed-tensors",
            "format": "mxfp4-pack-quantized",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "float",
                        "strategy": "group",
                        "group_size": 32,
                        "symmetric": true
                    }
                }
            },
            "ignore": [
                "model.language_model.embed_tokens",
                "model.language_model.layers.0.input_layernorm",
                "model.language_model.layers.0.linear_attn.conv1d",
                "model.language_model.layers.0.linear_attn.in_proj_a",
                "model.language_model.norm",
                "model.visual.blocks.0.attn.proj",
                "mtp.fc"
            ]
        }"#;
        let mut cfg: QuantConfig = serde_json::from_str(json).unwrap();
        cfg.normalize_compressed_tensors();
        assert_eq!(cfg.quant_method, "mxfp4");
        assert_eq!(cfg.group_size, 32);
        assert_eq!(cfg.bits, 4);
        assert!(cfg.should_skip_module("model.language_model.embed_tokens"));
        assert!(cfg.should_skip_module("model.language_model.layers.0.linear_attn.conv1d"));
        assert!(!cfg.should_skip_module("model.language_model.layers.0.mlp.up_proj"));
    }

    #[test]
    fn test_kaitchup_27b_config() {
        let json = r#"{
            "config_groups": {
                "group_0": {
                    "format": "mxfp4-pack-quantized",
                    "input_activations": null,
                    "output_activations": null,
                    "targets": ["Linear"],
                    "weights": {
                        "actorder": null,
                        "block_structure": null,
                        "dynamic": false,
                        "group_size": 32,
                        "num_bits": 4,
                        "observer": "memoryless_minmax",
                        "observer_kwargs": {},
                        "scale_dtype": "torch.uint8",
                        "strategy": "group",
                        "symmetric": true,
                        "type": "float",
                        "zp_dtype": null
                    }
                }
            },
            "format": "mxfp4-pack-quantized",
            "global_compression_ratio": null,
            "ignore": [
                "model.visual.blocks.0.attn.qkv",
                "model.language_model.layers.0.linear_attn.out_proj",
                "lm_head"
            ],
            "kv_cache_scheme": null,
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed",
            "sparsity_config": {},
            "transform_config": {},
            "version": "0.13.1.dev53+gd96634b"
        }"#;
        let mut cfg: QuantConfig = serde_json::from_str(json).unwrap();
        cfg.normalize_compressed_tensors();
        assert_eq!(cfg.quant_method, "mxfp4");
        assert_eq!(cfg.group_size, 32);
        assert_eq!(cfg.bits, 4);
        assert!(cfg.should_skip_module("model.language_model.layers.0.linear_attn.out_proj"));
        assert!(cfg.should_skip_module("lm_head"));
    }

    #[test]
    fn test_122b_regex_config() {
        let json = r#"{
            "quant_method": "compressed-tensors",
            "format": "mxfp4-pack-quantized",
            "quantization_status": "compressed",
            "config_groups": {
                "group_0": {
                    "format": "mxfp4-pack-quantized",
                    "weights": {
                        "num_bits": 4,
                        "type": "float",
                        "strategy": "group",
                        "group_size": 32,
                        "symmetric": true,
                        "scale_dtype": "torch.uint8",
                        "dynamic": false,
                        "actorder": null,
                        "block_structure": null,
                        "observer": "minmax",
                        "observer_kwargs": {},
                        "zp_dtype": null
                    },
                    "targets": ["Linear"],
                    "input_activations": null,
                    "output_activations": null
                }
            },
            "ignore": [
                "re:.*self_attn.*",
                "re:.*linear_attn.*",
                "re:.*.mlp.gate$",
                "re:.*shared_expert_gate.*",
                "re:.*lm_head.*",
                "re:.*embed_tokens.*",
                "re:.*visual.*",
                "re:.*mtp.*"
            ],
            "kv_cache_scheme": null,
            "sparsity_config": {},
            "transform_config": {}
        }"#;
        let mut cfg: QuantConfig = serde_json::from_str(json).unwrap();
        cfg.normalize_compressed_tensors();
        assert_eq!(cfg.quant_method, "mxfp4");
        assert_eq!(cfg.group_size, 32);
        assert_eq!(cfg.bits, 4);
        assert!(cfg.should_skip_module("model.layers.5.self_attn.q_proj"));
        assert!(cfg.should_skip_module("model.layers.5.linear_attn.out_proj"));
        assert!(cfg.should_skip_module("model.layers.3.mlp.gate"));
        assert!(!cfg.should_skip_module("model.layers.3.mlp.gate_proj"));
        assert!(cfg.should_skip_module("model.layers.3.shared_expert_gate"));
        assert!(cfg.should_skip_module("lm_head"));
        assert!(cfg.should_skip_module("model.embed_tokens"));
        assert!(cfg.should_skip_module("model.visual.blocks.0.attn.qkv"));
        assert!(cfg.should_skip_module("mtp.fc"));
        assert!(!cfg.should_skip_module("model.layers.3.mlp.up_proj"));
        assert!(!cfg.should_skip_module("model.layers.3.mlp.down_proj"));
    }

    #[test]
    fn test_2imi9_9b_config() {
        let json = r#"{
            "config_groups": {
                "group_0": {
                    "format": "mxfp4-pack-quantized",
                    "input_activations": null,
                    "output_activations": null,
                    "targets": ["Linear"],
                    "weights": {
                        "actorder": null,
                        "block_structure": null,
                        "dynamic": false,
                        "group_size": 32,
                        "num_bits": 4,
                        "observer": "memoryless_minmax",
                        "observer_kwargs": {},
                        "scale_dtype": "torch.uint8",
                        "strategy": "group",
                        "symmetric": true,
                        "type": "float",
                        "zp_dtype": null
                    }
                }
            },
            "format": "mxfp4-pack-quantized",
            "global_compression_ratio": null,
            "ignore": [
                "model.layers.0.linear_attn.out_proj",
                "model.layers.0.linear_attn.in_proj_qkv",
                "lm_head"
            ],
            "kv_cache_scheme": null,
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed",
            "sparsity_config": {},
            "transform_config": {},
            "version": "0.14.1.a20260310"
        }"#;
        let mut cfg: QuantConfig = serde_json::from_str(json).unwrap();
        cfg.normalize_compressed_tensors();
        assert_eq!(cfg.quant_method, "mxfp4");
        assert_eq!(cfg.group_size, 32);
        assert_eq!(cfg.bits, 4);
        assert!(cfg.should_skip_module("model.layers.0.linear_attn.out_proj"));
        assert!(cfg.should_skip_module("lm_head"));
        assert!(!cfg.should_skip_module("model.layers.1.mlp.up_proj"));
    }

    #[test]
    fn test_nvfp4_axionml_4b_config() {
        let json = r#"{
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "config_groups": {
                "group_0": {
                    "input_activations": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16},
                    "weights": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16},
                    "targets": ["Linear"]
                }
            },
            "ignore": [
                "lm_head",
                "model.language_model.layers.0.linear_attn.conv1d",
                "model.visual*",
                "mtp.layers.0*"
            ]
        }"#;
        let mut cfg: QuantConfig = serde_json::from_str(json).unwrap();
        cfg.normalize_compressed_tensors();
        assert_eq!(cfg.quant_method, "nvfp4");
        assert_eq!(cfg.group_size, 16);
        assert_eq!(cfg.bits, 4);
        assert!(cfg.should_skip_module("lm_head"));
        assert!(cfg.should_skip_module("model.visual.encoder.layers.0.self_attn"));
        assert!(cfg.should_skip_module("mtp.layers.0.mlp.gate_proj"));
        assert!(!cfg.should_skip_module("model.language_model.layers.1.mlp.up_proj"));
    }

    #[test]
    fn test_nvfp4_glob_wildcards() {
        let json = r#"{
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "ignore": [
                "lm_head",
                "*.mlp.shared_expert.*",
                "model.layers.0.self_attn*",
                "model.layers.92*"
            ]
        }"#;
        let mut cfg: QuantConfig = serde_json::from_str(json).unwrap();
        cfg.normalize_compressed_tensors();
        assert_eq!(cfg.quant_method, "nvfp4");
        assert!(cfg.should_skip_module("lm_head"));
        assert!(cfg.should_skip_module("model.layers.5.mlp.shared_expert.gate_proj"));
        assert!(cfg.should_skip_module("model.layers.0.self_attn.q_proj"));
        assert!(cfg.should_skip_module("model.layers.0.self_attn.k_proj"));
        assert!(cfg.should_skip_module("model.layers.92.self_attn.q_proj"));
        assert!(!cfg.should_skip_module("model.layers.1.self_attn.q_proj"));
        assert!(!cfg.should_skip_module("model.layers.5.mlp.gate_proj"));
    }

    #[test]
    fn test_mlx_nvfp4_normalized() {
        // MLX-community models use U32-packed weights with FP8 E4M3 scales.
        // They should normalize to "nvfp4" with is_mlx_nvfp4 = true, and
        // the weight loader repacks U32 → U8 at load time.
        let json = r#"{
            "group_size": 16,
            "bits": 4,
            "mode": "nvfp4"
        }"#;
        let mut cfg: QuantConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.quant_method, "");
        cfg.normalize_compressed_tensors();
        assert_eq!(cfg.quant_method, "nvfp4");
        assert!(cfg.is_mlx_nvfp4, "MLX mode=nvfp4 must set is_mlx_nvfp4");
        assert_eq!(cfg.group_size, 16);
        assert_eq!(cfg.bits, 4);
    }

    #[test]
    fn test_nvfp4_compressed_tensors_format() {
        // RedHatAI/Qwen3.5-122B-A10B-NVFP4 style: compressed-tensors + nvfp4-pack-quantized
        let json = r#"{
            "quant_method": "compressed-tensors",
            "format": "nvfp4-pack-quantized",
            "config_groups": {
                "group_0": {
                    "format": "nvfp4-pack-quantized",
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "float",
                        "group_size": 16,
                        "strategy": "tensor_group",
                        "symmetric": true,
                        "dynamic": false,
                        "scale_dtype": "torch.float8_e4m3fn"
                    },
                    "input_activations": {
                        "num_bits": 4,
                        "type": "float",
                        "group_size": 16,
                        "dynamic": "local",
                        "scale_dtype": "torch.float8_e4m3fn"
                    }
                }
            },
            "ignore": [
                "lm_head",
                "model.visual.blocks.0.attn.qkv",
                "model.language_model.layers.0.linear_attn.out_proj",
                "model.language_model.layers.0.mlp.gate",
                "model.language_model.layers.0.mlp.shared_expert_gate"
            ]
        }"#;
        let mut cfg: QuantConfig = serde_json::from_str(json).unwrap();
        cfg.normalize_compressed_tensors();
        assert_eq!(cfg.quant_method, "nvfp4");
        assert_eq!(cfg.bits, 4);
        assert_eq!(cfg.group_size, 16);
        assert!(cfg.should_skip_module("lm_head"));
        assert!(cfg.should_skip_module("model.visual.blocks.0.attn.qkv"));
        assert!(cfg.should_skip_module("model.language_model.layers.0.linear_attn.out_proj"));
        assert!(cfg.should_skip_module("model.language_model.layers.0.mlp.gate"));
        assert!(cfg.should_skip_module("model.language_model.layers.0.mlp.shared_expert_gate"));
        assert!(!cfg.should_skip_module("model.language_model.layers.0.mlp.gate_proj"));
        assert!(!cfg.should_skip_module("model.language_model.layers.0.mlp.down_proj"));
    }

    #[test]
    fn test_nvfp4_compressed_tensors_detect_from_groups() {
        // compressed-tensors without top-level format, detected from config_groups
        let json = r#"{
            "quant_method": "compressed-tensors",
            "config_groups": {
                "group_0": {
                    "format": "nvfp4-pack-quantized",
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "float",
                        "group_size": 16
                    }
                }
            }
        }"#;
        let mut cfg: QuantConfig = serde_json::from_str(json).unwrap();
        cfg.normalize_compressed_tensors();
        assert_eq!(cfg.quant_method, "nvfp4");
        assert_eq!(cfg.bits, 4);
        assert_eq!(cfg.group_size, 16);
    }

    #[test]
    fn test_mixed_precision_nvfp4_fp8_from_quantized_layers() {
        let json = r#"{
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "config_groups": {
                "group_0": {
                    "input_activations": {"dynamic": false, "num_bits": 8, "type": "float"},
                    "weights": {"dynamic": false, "num_bits": 8, "type": "float"},
                    "targets": ["model.layers.0.linear_attn.in_proj_qkv"]
                },
                "group_1": {
                    "input_activations": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16},
                    "weights": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16},
                    "targets": ["model.layers.0.mlp.experts"]
                }
            },
            "quantized_layers": {
                "model.layers.0.linear_attn.in_proj_qkv": {"quant_algo": "FP8"},
                "model.layers.0.mlp.experts": {"quant_algo": "W4A16_NVFP4", "group_size": 16}
            },
            "ignore": ["mtp*"]
        }"#;
        let mut cfg: QuantConfig = serde_json::from_str(json).unwrap();
        cfg.normalize_compressed_tensors();
        assert_eq!(cfg.quant_method, "nvfp4");
        assert_eq!(cfg.group_size, 16);
        assert_eq!(cfg.bits, 4);
    }

    #[test]
    fn test_mixed_precision_fp8_only_from_quantized_layers() {
        let json = r#"{
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {
                "model.layers.0.self_attn.q_proj": {"quant_algo": "FP8"},
                "model.layers.0.self_attn.k_proj": {"quant_algo": "FP8"}
            },
            "ignore": ["mtp*"]
        }"#;
        let mut cfg: QuantConfig = serde_json::from_str(json).unwrap();
        cfg.normalize_compressed_tensors();
        assert_eq!(cfg.quant_method, "fp8");
    }

    #[test]
    fn test_modelopt_fp8_normalization() {
        let json = r#"{
            "quant_method": "modelopt",
            "quant_algo": "FP8"
        }"#;
        let mut cfg: QuantConfig = serde_json::from_str(json).unwrap();
        cfg.normalize_compressed_tensors();
        assert_eq!(cfg.quant_method, "fp8");
    }
}
