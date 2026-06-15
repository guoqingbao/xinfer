// src/utils/metrics/prometheus.rs
//! Prometheus exporter module for vllm.rs instrumentation
//!
//! Provides Prometheus-compatible metrics endpoint at /metrics
//! by using metrics-exporter-prometheus which handles the metrics crate.

use metrics_exporter_prometheus::PrometheusBuilder;
use std::sync::Arc;

/// Re-export PrometheusHandle for use in other modules
pub use metrics_exporter_prometheus::PrometheusHandle;

/// Global Prometheus handle storage - initialized once at startup
static PROMETHEUS_HANDLE: std::sync::OnceLock<Arc<PrometheusHandle>> = std::sync::OnceLock::new();

/// Initialize Prometheus exporter - installs the metrics recorder as global
/// This MUST be called before any metrics are recorded
/// Panics if called twice (double initialization is a fatal error)
pub fn init_prometheus() -> Arc<PrometheusHandle> {
    // Use install_recorder which both installs the recorder and returns the handle
    let handle = PrometheusBuilder::new()
        .install_recorder()
        .expect("Failed to install Prometheus recorder - this is a fatal configuration error");

    // Store the handle for later use
    let handle_arc = Arc::new(handle);

    // Install the handle in the global OnceLock
    // This MUST succeed on the first call - if it fails, we have a programming error
    match PROMETHEUS_HANDLE.set(handle_arc.clone()) {
        Ok(()) => (),
        Err(_) => panic!("Prometheus handle already set - init_prometheus() called more than once"),
    };

    handle_arc
}

/// Get the Prometheus handle for rendering metrics
/// This is called from the /metrics endpoint handler
pub fn get_prometheus_handle() -> &'static Arc<PrometheusHandle> {
    PROMETHEUS_HANDLE.get().expect("Prometheus handle not initialized - call init_prometheus() before starting the server")
}

/// Initialize Prometheus exporter - creates and installs the recorder
/// This is a convenience wrapper around init_prometheus()
pub fn initialize() -> Arc<PrometheusHandle> {
    init_prometheus()
}

/// Render metrics using the Prometheus handle
/// This is called from the /metrics endpoint handler
pub async fn render_metrics() -> String {
    // Get the handle - panics if not initialized (which is a fatal error)
    let handle = get_prometheus_handle();
    // Render the metrics - this always returns a string (possibly empty if no metrics)
    handle.render()
}