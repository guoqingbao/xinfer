// src/utils/metrics/mod.rs
//! Metrics module for vllm.rs instrumentation
//!
//! Provides comprehensive GPU and system metrics collection using nvml-wrapper
//! and sysinfo with a DRY macro-based approach.
//!
//! # Feature Gates
//!
//! - `cuda`: Enables GPU metrics via nvml-wrapper (optional)
//! - All other metrics are ALWAYS enabled

pub mod record;
pub mod nvml;
pub mod cpu;
pub mod metrics_macros;
pub mod prometheus;
pub mod performance_curve;

// Re-export all modules
pub use record::*;
pub use nvml::{collect_device_metrics, collect_all_metrics};
pub use cpu::{collect_system_metrics, get_metrics_interval, start_cpu_monitor};
pub use prometheus::{init_prometheus, get_prometheus_handle, render_metrics, initialize};
pub use performance_curve::{WorkloadType, BottleneckType, PerformancePoint, PerformanceCurve, PerformanceCurveManager, BinarySearchOptimizer, init_performance_curve_manager, init_binary_search_optimizer, get_performance_curve_manager, get_binary_search_optimizer, record_performance_point, record_binary_search_performance, get_binary_search_optimal_batch_size, get_binary_search_current_test_value, binary_search_has_converged, reset_binary_search_optimizer};