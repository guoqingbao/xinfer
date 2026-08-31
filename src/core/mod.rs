pub mod block_manager;
pub mod engine;
pub mod prefix_cache;
pub mod runner;
pub mod scheduler;
pub mod sequence;
#[cfg(feature = "python")]
use pyo3::pyclass;

#[cfg(feature = "python")]
#[pyclass]
#[derive(Debug, Clone)]
pub struct GenerationOutput {
    #[pyo3(get)]
    pub seq_id: usize,
    #[pyo3(get)]
    pub prompt_length: usize,
    #[pyo3(get)]
    pub prompt_start_time: usize,
    #[pyo3(get)]
    pub decode_start_time: usize,
    #[pyo3(get)]
    pub decode_finish_time: usize,
    #[pyo3(get)]
    pub decoded_length: usize,
    #[pyo3(get)]
    pub decode_output: String,
    #[pyo3(get)]
    pub stop_sequence: Option<String>,
}

#[cfg(not(feature = "python"))]
#[derive(Debug, Clone)]
pub struct GenerationOutput {
    pub seq_id: usize,
    pub prompt_length: usize,
    pub prompt_start_time: usize,
    pub decode_start_time: usize,
    pub decode_finish_time: usize,
    pub decoded_length: usize,
    pub decode_output: String,
    pub stop_sequence: Option<String>,
}

#[macro_export]
macro_rules! log_info {
    ($($arg:tt)*) => {
        {
            #[cfg(feature = "python")]
            {
                use colored::Colorize;
                let s = format!($($arg)*);
                println!("{}", String::from(s).truecolor(211, 211, 211));
            }
            #[cfg(not(feature = "python"))]
            {
                tracing::info!($($arg)*);
            }
        }
    };
}

#[macro_export]
macro_rules! log_warn {
    ($($arg:tt)*) => {
        {
            #[cfg(feature = "python")]
            {
                use colored::Colorize;
                let s = format!($($arg)*);
                eprintln!("{}", String::from(s).yellow());
            }
            #[cfg(not(feature = "python"))]
            {
                tracing::warn!($($arg)*);
            }
        }
    };
}

#[macro_export]
macro_rules! log_error {
    ($($arg:tt)*) => {
        {
            #[cfg(feature = "python")]
            {
                use colored::Colorize;
                let s = format!($($arg)*);
                eprintln!("{}", String::from(s).red());
            }
            #[cfg(not(feature = "python"))]
            {
                tracing::error!($($arg)*);
            }
        }
    };
}

pub trait DecodeStreamTrait: Send + Sync {
    fn step(&mut self, id: u32) -> candle_core::Result<Option<String>>;
}

struct StreamWithTokenizer<M, N, PT, PP, D>
where
    M: tokenizers::Model + Send + Sync + 'static,
    N: tokenizers::Normalizer + Send + Sync + 'static,
    PT: tokenizers::PreTokenizer + Send + Sync + 'static,
    PP: tokenizers::PostProcessor + Send + Sync + 'static,
    D: tokenizers::Decoder + Send + Sync + 'static,
{
    _tokenizer: Box<tokenizers::TokenizerImpl<M, N, PT, PP, D>>,
    stream: tokenizers::DecodeStream<'static, M, N, PT, PP, D>,
}

impl<M, N, PT, PP, D> DecodeStreamTrait for StreamWithTokenizer<M, N, PT, PP, D>
where
    M: tokenizers::Model + Send + Sync + 'static,
    N: tokenizers::Normalizer + Send + Sync + 'static,
    PT: tokenizers::PreTokenizer + Send + Sync + 'static,
    PP: tokenizers::PostProcessor + Send + Sync + 'static,
    D: tokenizers::Decoder + Send + Sync + 'static,
{
    fn step(&mut self, id: u32) -> candle_core::Result<Option<String>> {
        self.stream.step(id).map_err(candle_core::Error::wrap)
    }
}

type DecodeStreamType = Box<dyn DecodeStreamTrait + Send + Sync>;

#[macro_export]
macro_rules! build_model {
    ($model_type:expr, $vb:expr, $comm:expr, $config:expr, $dtype:expr, $is_rope_i:expr, $device:expr, $reporter:expr,
        { $( $variant:ident => $ctor:ident ),+ $(,)? }
    ) => {{
        match $model_type {
            $( ModelType::$variant => Ok::<Model, candle_core::Error>(Model::$variant(Arc::new($ctor::new(
                $vb,
                $comm.clone(),
                $config,
                $dtype,
                $is_rope_i,
                $device,
                Arc::clone(&$reporter),
            )?))), )+
            _ => {
                candle_core::bail!("Unsupported model type: {:?}", $model_type);
            }
        }
    }};
}

#[macro_export]
macro_rules! model_call {
    ($model:expr, $method:ident,
        ($input_ids:expr, $positions:expr, $kv:expr, $input_metadata:expr),
        { $( $variant:ident => $extra:expr ),+ $(,)? }
        $(, $fallback:expr )?
    ) => {{
        match $model {
            $( Model::$variant(model) => model.$method($input_ids, $positions, $kv, $input_metadata, $extra), )+
            $( _ => $fallback, )?
        }
    }};
}

#[cfg(all(feature = "cuda", feature = "graph"))]
#[macro_export]
macro_rules! graph_extra_arg {
    (EmbedInputs, $embeded_inputs:ident) => {
        $embeded_inputs
    };
    (NoneArg, $embeded_inputs:ident) => {
        None
    };
}

#[cfg(all(feature = "cuda", feature = "graph"))]
#[macro_export]
macro_rules! graph_wrapper {
    ($model:expr, $device:expr,
        { $( $variant:ident => $arg:tt ),+ $(,)? }
    ) => {{
        match $model {
            $( Model::$variant(m) => {
                let model_arc = Arc::clone(m);
                let closure = move |input_ids: &Tensor,
                                    positions: &Tensor,
                                    kv_caches: Option<&Vec<(Tensor, Tensor)>>,
                                    input_metadata: &InputMetadata,
                                    embeded_inputs: bool| {
                    model_arc.forward(
                        input_ids,
                        positions,
                        kv_caches,
                        input_metadata,
                        crate::graph_extra_arg!($arg, embeded_inputs),
                    )
                };
                let boxed_closure: Box<ModelFn> = Box::new(closure);
                CudaGraphWrapper::new(boxed_closure, $device.as_cuda_device()?.clone().into())
            }, )+
        }
    }};
}
