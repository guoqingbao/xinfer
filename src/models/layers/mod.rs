pub mod attention;
pub mod deepstack;
pub mod deltanet;
pub mod distributed;
pub mod indexer;
pub mod linear;
pub mod mask;
pub mod mla_attention;
pub mod mlp;
pub mod moe;
pub mod others;
pub mod rotary_emb;
pub mod wna16;
use crate::utils::downloader::ModelPaths;
use crate::utils::gguf_varbuilder::VarBuilder as QVarBuilder;
use candle_core::quantized::GgmlDType;
use candle_core::DType;
use candle_core::{Device, Result, Tensor};
use candle_nn::var_builder::ShardedVarBuilder as VarBuilder;
use either::Either;
use std::collections::HashMap;
use std::path::Path;

pub fn collect_key_map<'a, const N: usize>(
    is_qvar_builder: bool,
    pairs: [(&'a str, &'a str); N],
) -> HashMap<&'a str, &'a str> {
    if is_qvar_builder {
        pairs.into_iter().collect()
    } else {
        pairs.into_iter().map(|(key, _)| (key, key)).collect()
    }
}

pub fn isq_high_precision_quant(quant: &str) -> &'static str {
    match quant {
        "q8" | "q80" | "q8_0" => "q8_0",
        _ => "q6k",
    }
}

pub fn isq_high_precision_dtype(quant: Option<&str>) -> GgmlDType {
    match quant {
        Some("q8" | "q80" | "q8_0") => GgmlDType::Q8_0,
        _ => GgmlDType::Q6K,
    }
}

#[derive(Clone)]
pub struct VarBuilderX<'a>(
    pub Either<VarBuilder<'a>, QVarBuilder>,
    pub String,
    pub Option<Either<VarBuilder<'a>, QVarBuilder>>,
    pub Option<VarBuilder<'a>>,
    pub Option<Vec<std::path::PathBuf>>,
);

impl VarBuilderX<'_> {
    pub fn from_gguf_file<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self> {
        let file_path = path.as_ref().to_path_buf();
        let vb = crate::utils::gguf_varbuilder::VarBuilder::from_gguf(path, device)?;
        Ok(Self(
            Either::Right(vb),
            String::new(),
            None,
            None,
            Some(vec![file_path]),
        ))
    }

    pub fn new(
        model_pathes: &ModelPaths,
        is_gguf: bool,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        assert!(
            !model_pathes.get_weight_filenames().is_empty(),
            "No weight files found!"
        );
        let weight_files = model_pathes.get_weight_filenames();
        if is_gguf {
            let vb =
                crate::utils::gguf_varbuilder::VarBuilder::from_gguf_files(&weight_files, device)?;
            let auxiliary_vb = model_pathes
                .get_auxiliary_filenames()
                .first()
                .map(|path| crate::utils::gguf_varbuilder::VarBuilder::from_gguf(path, device))
                .transpose()?
                .map(Either::Right);
            Ok(Self(
                Either::Right(vb),
                String::new(),
                auxiliary_vb,
                None,
                Some(weight_files.clone()),
            ))
        } else {
            let vb = unsafe {
                candle_nn::var_builder::ShardedSafeTensors::var_builder(
                    &weight_files,
                    dtype,
                    device,
                )?
            };
            let cpu_vb = if matches!(device, Device::Cpu) {
                None
            } else {
                Some(unsafe {
                    candle_nn::var_builder::ShardedSafeTensors::var_builder(
                        &weight_files,
                        dtype,
                        &Device::Cpu,
                    )?
                })
            };
            Ok(Self(
                Either::Left(vb),
                String::new(),
                None,
                cpu_vb,
                Some(weight_files),
            ))
        }
    }

    pub fn is_var_builder(&self) -> bool {
        matches!(self.0, Either::Left(_))
    }

    pub fn is_qvar_builder(&self) -> bool {
        matches!(self.0, Either::Right(_))
    }

    pub fn device(&self) -> Device {
        match &self.0 {
            Either::Left(vb) => vb.device().clone(),
            Either::Right(vb) => vb.device().clone(),
        }
    }

    pub fn pp(&self, name: &str) -> VarBuilderX<'_> {
        let next_path = if self.1.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.1, name)
        };
        let cpu_vb = self.3.as_ref().map(|vb| vb.pp(name));
        match &self.0 {
            Either::Left(vb) => VarBuilderX(
                Either::Left(vb.pp(name)),
                next_path,
                self.2.clone(),
                cpu_vb,
                self.4.clone(),
            ),
            Either::Right(vb) => VarBuilderX(
                Either::Right(vb.pp(name)),
                next_path,
                self.2.clone(),
                cpu_vb,
                self.4.clone(),
            ),
        }
    }

    pub fn aux(&self) -> Option<VarBuilderX<'_>> {
        self.2
            .as_ref()
            .cloned()
            .map(|vb| VarBuilderX(vb, String::new(), None, None, None))
    }

    pub fn gguf_path(&self) -> Option<&str> {
        match &self.0 {
            Either::Right(vb) => Some(vb.gguf_path()),
            _ => None,
        }
    }

    pub fn weight_paths(&self) -> Option<Vec<std::path::PathBuf>> {
        self.4.clone()
    }

    pub fn cpu_var_builder(&self) -> Option<VarBuilder<'_>> {
        self.3.clone()
    }

    pub fn module_path(&self) -> &str {
        &self.1
    }

    pub fn has_key(&self, name: &str) -> bool {
        match &self.0 {
            Either::Left(vb) => vb.contains_tensor(name),
            Either::Right(vb) => vb.contains_key(name),
        }
    }

    pub fn tensor_shape(&self, name: &str) -> Option<Vec<usize>> {
        match &self.0 {
            Either::Left(_) => None,
            Either::Right(vb) => vb.tensor_shape(name),
        }
    }

    pub fn get_with_hints_dtype<S: Into<candle_core::Shape>>(
        &self,
        s: S,
        name: &str,
        shard: candle_nn::var_builder::Shard,
        dtype: DType,
    ) -> Result<Tensor> {
        match &self.0 {
            Either::Left(vb) => vb.get_with_hints_dtype(s, name, shard, dtype),
            Either::Right(vb) => vb.get(s, name).and_then(|x| x.dequantize(vb.device())),
        }
    }

    pub fn get<S: Into<candle_core::Shape>>(&self, s: S, name: &str) -> Result<Tensor> {
        match &self.0 {
            Either::Left(vb) => vb.get(s, name),
            Either::Right(vb) => vb.get(s, name).and_then(|x| x.dequantize(vb.device())),
        }
    }
}
