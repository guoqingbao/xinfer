//! Runtime capability diagnostics for SM70 / V100 deployments.
//!
//! Modern quantized checkpoints (NVFP4, FP8) rely on hardware tensor-core
//! paths that only exist from Hopper (SM90) and Blackwell (SM100) onwards.
//! On Volta (SM70) the same checkpoints fall back to software dequantization
//! or reference recurrences, which is often *slower* than a plain F16 model
//! while saving nothing but memory.
//!
//! This module records, per acceleration path, whether it is actually
//! enabled and why it is not when it is disabled. The report is printed at
//! startup and exposed over HTTP so an operator can tell "my model is slow"
//! from "my model took a slow path".
//!
//! The design mirrors the acceleration-report surface of 1CatAI/1Cat-vLLM,
//! which ships an equivalent `/v1/sm70/acceleration` endpoint.

use std::fmt;

/// One acceleration path and its current state.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct CapabilityEntry {
    /// Stable identifier, e.g. `nvfp4_hardware_gemm`.
    pub name: &'static str,
    /// Whether the accelerated path is active.
    pub enabled: bool,
    /// Why the path is disabled. `None` when enabled.
    pub reason: Option<String>,
}

impl CapabilityEntry {
    fn enabled(name: &'static str) -> Self {
        Self {
            name,
            enabled: true,
            reason: None,
        }
    }

    fn disabled(name: &'static str, reason: impl Into<String>) -> Self {
        Self {
            name,
            enabled: false,
            reason: Some(reason.into()),
        }
    }
}

/// Full acceleration report for one running engine.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct AccelerationReport {
    /// Compute capability of the device, e.g. `70` for V100.
    pub compute_capability: Option<usize>,
    /// Model architecture family, e.g. `Qwen3_5ForConditionalGeneration`.
    pub architecture: Option<String>,
    /// Whether the model weights use a 4-bit checkpoint format.
    pub uses_nvfp4_weights: bool,
    /// Entries in report order.
    pub entries: Vec<CapabilityEntry>,
}

impl AccelerationReport {
    /// Number of acceleration paths that are active.
    pub fn enabled_count(&self) -> usize {
        self.entries.iter().filter(|e| e.enabled).count()
    }

    /// Paths that are expected on this hardware but are not active.
    pub fn degraded(&self) -> impl Iterator<Item = &CapabilityEntry> {
        self.entries.iter().filter(|e| !e.enabled)
    }

    /// True when an NVFP4 checkpoint is running on a device without hardware
    /// FP4 tensor cores, i.e. every NVFP4 GEMM takes the software
    /// dequantization path.
    ///
    /// On such a device NVFP4 is usually a net loss: the weights occupy a
    /// quarter of the memory, but the GEMM does strictly more work than a
    /// plain F16 GEMM because each 4-bit weight is expanded at runtime.
    pub fn nvfp4_is_degraded(&self) -> bool {
        self.uses_nvfp4_weights
            && self
                .compute_capability
                .map(|cc| cc < 100)
                .unwrap_or(false)
    }

    /// Multi-line human readable summary, one line per path.
    pub fn summary(&self) -> String {
        let mut out = String::new();
        if let Some(cc) = self.compute_capability {
            out.push_str(&format!(
                "Device: SM{cc} | architecture: {} | nvfp4 weights: {}\n",
                self.architecture.as_deref().unwrap_or("unknown"),
                self.uses_nvfp4_weights
            ));
        }
        for entry in &self.entries {
            match (&entry.enabled, &entry.reason) {
                (true, _) => out.push_str(&format!("  [ok]    {}\n", entry.name)),
                (false, Some(reason)) => {
                    out.push_str(&format!("  [slow]  {} — {}\n", entry.name, reason))
                }
                (false, None) => out.push_str(&format!("  [slow]  {}\n", entry.name)),
            }
        }
        out
    }
}

impl fmt::Display for AccelerationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

/// Environment switch that turns an unexpectedly slow deployment into a
/// startup failure.
pub const REQUIRE_ACCELERATION_ENV: &str = "XINFER_REQUIRE_ACCELERATION";

/// Whether the operator asked for a hard failure when an expected
/// acceleration path is missing.
pub fn require_acceleration() -> bool {
    match std::env::var(REQUIRE_ACCELERATION_ENV) {
        Ok(v) => matches!(v.trim(), "1" | "true" | "yes"),
        Err(_) => false,
    }
}

/// Inputs needed to evaluate the acceleration paths.
pub struct ReportInputs<'a> {
    /// Compute capability of the target device (e.g. 70 for V100, 90 for H100).
    pub compute_capability: Option<usize>,
    /// Model architecture family.
    pub architecture: String,
    /// Whether the checkpoint stores 4-bit (NVFP4/MXFP4) weights.
    pub uses_nvfp4_weights: bool,    /// Whether the build enables the FlashInfer backend.
    pub flashinfer_built: bool,
    /// Whether the FlashInfer backend was actually initialized.
    pub flashinfer_active: bool,
    /// Why FlashInfer was skipped, when it was.
    pub flashinfer_skip_reason: Option<&'a str>,
    /// Whether the build enables CUTLASS kernels.
    pub cutlass_built: bool,
    /// Whether the loaded model actually dispatches to CUTLASS kernels.
    pub cutlass_active: bool,
    /// Whether the build enables CUDA graph capture.
    pub cuda_graph_built: bool,
    /// Whether CUDA graph capture succeeded.
    pub cuda_graph_active: bool,
    /// Whether the GDN (linear attention) prefill uses the fast chunked path.
    pub gdn_fast_prefill: bool,
    /// Why the GDN prefill uses the reference path.
    pub gdn_prefill_reason: Option<&'a str>,
    /// KV cache dtype in use.
    pub kv_cache_dtype: &'a str,
}

impl ReportInputs<'_> {
    /// Build the inputs from the engine configuration and model config.
    ///
    /// This is the single place that derives the report facts, so the runner
    /// (which owns the runtime decisions) and the engine (which serves the
    /// HTTP report) cannot drift apart.
    pub fn from_engine(
        config: &crate::utils::config::Config,
        econfig: &crate::utils::config::EngineConfig,
        compute_capability: Option<usize>,
    ) -> Self {
        let architecture = config
            .architectures
            .as_ref()
            .and_then(|archs| archs.first())
            .cloned()
            .unwrap_or_else(|| "unknown".to_string());

        let uses_nvfp4_weights = config
            .quantization_config
            .as_ref()
            .map(|q| matches!(q.quant_method.as_str(), "nvfp4" | "mxfp4"))
            .unwrap_or(false);

        // The CUTLASS kernels only apply to FP8 / FP4 linear layers. A
        // pure-F16 model never dispatches to them, so reporting them as
        // active would be wrong and reserving their workspace wastes memory.
        let cutlass_built = cfg!(feature = "cutlass");
        let cutlass_active = cutlass_built
            && matches!(
                config
                    .quantization_config
                    .as_ref()
                    .map(|q| q.quant_method.as_str()),
                Some("fp8") | Some("nvfp4") | Some("mxfp4")
            );

        let cuda_graph_built = cfg!(all(feature = "cuda", feature = "graph"));
        let cuda_graph_active = cuda_graph_built && !econfig.disable_cuda_graph;

        // FlashInfer is only usable for a narrow set of KV dtypes, head
        // layouts and architectures. The runner performs the authoritative
        // check; the engine mirrors the same rules so the report is stable
        // even when the runner lives in another process.
        let flashinfer_built = cfg!(feature = "flashinfer");
        let gqa_supported = crate::utils::kvcache_allocator::flashinfer_gqa_group_supported(
            econfig, config,
        );

        let flashinfer_skip_reason: Option<&'static str> = if flashinfer_built {
            if config.kvcache_dtype.is_turboquant() {
                Some("TurboQuant KV cache is not supported by FlashInfer")
            } else if config.kvcache_dtype.is_fp8_keys()
                && !attention_rs::has_flashinfer_fp8_e4m3()
            {
                Some("the build has no FlashInfer FP8 E4M3 support")
            } else if !gqa_supported {
                Some("the GQA group size is not supported by the FlashInfer kernels")
            } else if config
                .architectures
                .as_ref()
                .and_then(|a| a.first())
                .map(|a| {
                    a == "DeepseekV4ForCausalLM" || a == "deepseek_v4" || a == "deepseek4"
                })
                .unwrap_or(false)
            {
                Some("DeepSeek V4 does not consume FlashInfer attention")
            } else {
                None
            }
        } else {
            Some("build lacks the `flashinfer` feature")
        };

        // The chunked GDN prefill kernel is SM90-only.
        let gdn_fast_prefill = cfg!(feature = "flashinfer")
            && compute_capability == Some(90)
            && crate::utils::env::sm90_lower_precision_gdn_prefill();
        let gdn_prefill_reason: Option<&'static str> = if gdn_fast_prefill {
            None
        } else if compute_capability != Some(90) {
            Some("the chunked GDN prefill kernel requires SM90")
        } else {
            Some("the chunked GDN prefill kernel is not enabled")
        };

        Self {
            compute_capability,
            architecture,
            uses_nvfp4_weights,            flashinfer_built,
            flashinfer_active: flashinfer_built && flashinfer_skip_reason.is_none(),
            flashinfer_skip_reason,
            cutlass_built,
            cutlass_active,
            cuda_graph_built,
            cuda_graph_active,
            gdn_fast_prefill,
            gdn_prefill_reason,
            kv_cache_dtype: "see startup log",
        }
    }
}

/// Evaluate every acceleration path and build the report.
pub fn build_report(inputs: ReportInputs<'_>) -> AccelerationReport {
    let mut entries = Vec::new();

    // --- NVFP4 / FP4 GEMM -----------------------------------------------
    let cc = inputs.compute_capability;
    let hardware_fp4 = cc.map(|cc| cc >= 100).unwrap_or(false);
    if inputs.uses_nvfp4_weights {
        if hardware_fp4 {
            entries.push(CapabilityEntry::enabled("nvfp4_hardware_gemm"));
        } else {
            entries.push(CapabilityEntry::disabled(
                "nvfp4_hardware_gemm",
                format!(
                    "SM{} has no FP4 tensor cores; 4-bit weights are expanded to \
                     F16 at runtime (software dequantization), which is slower \
                     than a plain F16 GEMM and only saves memory",
                    cc.map(|c| c.to_string()).unwrap_or_else(|| "?".into())
                ),
            ));
        }
    }

    // --- FlashInfer attention -------------------------------------------
    if !inputs.flashinfer_built {
        entries.push(CapabilityEntry::disabled(
            "flashinfer_attention",
            "build lacks the `flashinfer` feature",
        ));
    } else if inputs.flashinfer_active {
        entries.push(CapabilityEntry::enabled("flashinfer_attention"));
    } else {
        entries.push(CapabilityEntry::disabled(
            "flashinfer_attention",
            inputs.flashinfer_skip_reason.unwrap_or("not initialized"),
        ));
    }

    // --- CUTLASS kernels --------------------------------------------------
    if !inputs.cutlass_built {
        entries.push(CapabilityEntry::disabled(
            "cutlass_kernels",
            "build lacks the `cutlass` feature",
        ));
    } else if inputs.cutlass_active {
        entries.push(CapabilityEntry::enabled("cutlass_kernels"));
    } else {
        entries.push(CapabilityEntry::disabled(
            "cutlass_kernels",
            "the loaded model does not dispatch to CUTLASS kernels",
        ));
    }

    // --- CUDA graph -------------------------------------------------------
    if !inputs.cuda_graph_built {
        entries.push(CapabilityEntry::disabled(
            "cuda_graph",
            "build lacks the `graph` feature",
        ));
    } else if inputs.cuda_graph_active {
        entries.push(CapabilityEntry::enabled("cuda_graph"));
    } else {
        entries.push(CapabilityEntry::disabled(
            "cuda_graph",
            "graph capture was skipped or failed",
        ));
    }

    // --- GDN / linear attention prefill -----------------------------------
    if inputs.gdn_fast_prefill {
        entries.push(CapabilityEntry::enabled("gdn_fast_prefill"));
    } else {
        entries.push(CapabilityEntry::disabled(
            "gdn_fast_prefill",
            inputs.gdn_prefill_reason.unwrap_or("reference recurrence"),
        ));
    }

    AccelerationReport {
        compute_capability: cc,
        architecture: Some(inputs.architecture.clone()),
        uses_nvfp4_weights: inputs.uses_nvfp4_weights,
        entries,
    }
}

/// Print the report and return the set of degraded paths.
///
/// When [`require_acceleration`] is set, a degraded NVFP4 hardware path is a
/// hard error: the operator explicitly asked to fail rather than silently
/// run a degraded configuration.
pub fn report_and_maybe_fail(report: &AccelerationReport) -> Result<(), String> {
    crate::log_info!("Acceleration report:\n{}", report.summary());

    if report.nvfp4_is_degraded() {
        crate::log_warn!(
            "NVFP4 weights are running on SM{} without hardware FP4 support. \
             Every NVFP4 GEMM expands 4-bit weights to F16 at runtime, which \
             is slower than a plain F16 GEMM; NVFP4 only saves memory here. \
             Consider an AWQ/GPTQ checkpoint for better throughput on this device.",
            report.compute_capability.unwrap_or(0)
        );
    }

    if require_acceleration() && report.nvfp4_is_degraded() {
        return Err(format!(
            "{} is set but the NVFP4 hardware GEMM path is unavailable on SM{}; \
             refusing to start a degraded configuration",
            REQUIRE_ACCELERATION_ENV,
            report.compute_capability.unwrap_or(0)
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_inputs<'a>(arch: &'a str) -> ReportInputs<'a> {
        ReportInputs {
            compute_capability: Some(70),
            architecture: arch.to_string(),
            uses_nvfp4_weights: false,
            flashinfer_built: false,
            flashinfer_active: false,
            flashinfer_skip_reason: None,
            cutlass_built: false,
            cutlass_active: false,
            cuda_graph_built: true,
            cuda_graph_active: true,
            gdn_fast_prefill: false,
            gdn_prefill_reason: Some("SM70 uses the reference recurrence"),
            kv_cache_dtype: "F16",
        }
    }

    #[test]
    fn nvfp4_on_v100_is_reported_as_degraded() {
        let mut inputs = base_inputs("Qwen3_5ForConditionalGeneration");
        inputs.uses_nvfp4_weights = true;
        let report = build_report(inputs);

        assert!(report.nvfp4_is_degraded());
        let entry = report
            .entries
            .iter()
            .find(|e| e.name == "nvfp4_hardware_gemm")
            .unwrap();
        assert!(!entry.enabled);
        assert!(entry.reason.as_ref().unwrap().contains("no FP4 tensor cores"));
    }

    #[test]
    fn nvfp4_on_blackwell_is_not_degraded() {
        let mut inputs = base_inputs("Qwen3_5ForConditionalGeneration");
        inputs.compute_capability = Some(100);
        inputs.uses_nvfp4_weights = true;
        let report = build_report(inputs);

        assert!(!report.nvfp4_is_degraded());
        let entry = report
            .entries
            .iter()
            .find(|e| e.name == "nvfp4_hardware_gemm")
            .unwrap();
        assert!(entry.enabled);
        assert!(entry.reason.is_none());
    }

    #[test]
    fn f16_model_on_v100_is_not_degraded() {
        let report = build_report(base_inputs("Qwen3ForCausalLM"));
        assert!(!report.nvfp4_is_degraded());
    }

    #[test]
    fn missing_flashinfer_records_the_skip_reason() {
        let mut inputs = base_inputs("Qwen3_5ForConditionalGeneration");
        inputs.flashinfer_built = true;
        inputs.flashinfer_skip_reason = Some("GQA group size 12 is unsupported");
        let report = build_report(inputs);

        let entry = report
            .entries
            .iter()
            .find(|e| e.name == "flashinfer_attention")
            .unwrap();
        assert!(!entry.enabled);
        assert_eq!(
            entry.reason.as_deref(),
            Some("GQA group size 12 is unsupported")
        );
    }

    #[test]
    fn enabled_count_matches_entries() {
        let report = build_report(base_inputs("Qwen3ForCausalLM"));
        assert_eq!(report.enabled_count(), report.entries.iter().filter(|e| e.enabled).count());
    }

    #[test]
    fn summary_lists_every_path() {
        let report = build_report(base_inputs("Qwen3ForCausalLM"));
        let summary = report.summary();
        for entry in &report.entries {
            assert!(summary.contains(entry.name), "missing {}", entry.name);
        }
    }
}
