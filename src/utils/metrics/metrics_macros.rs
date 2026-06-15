// src/utils/metrics/metrics_macros.rs
//! Metrics macros for DRY metric definitions
//!
//! This module provides macros for defining and recording metrics in a
//! consistent, DRY manner. All metric definitions are centralized and
//! auto-generated recording functions are created from the definitions.
//!
//! # Design Principles
//!
//! - **Single Source of Truth**: All metrics are defined in one place
//! - **Auto-Generation**: Recording functions are generated from definitions
//! - **Type Safety**: All metrics are strongly typed
//! - **Consistent Labels**: All metrics of the same type have consistent labels
//!
//! # Usage
//!
//! This module provides macros for defining and recording metrics.
//! See the individual macro documentation for usage examples.

// ============================================================================
// COUNTER MACROS
// ============================================================================

/// Define a counter metric
#[macro_export]
macro_rules! define_counter {
    ($name:ident, $description:expr, $labels:expr) => {
        pub const $name: MetricInfo = MetricInfo {
            name: stringify!($name),
            description: $description,
            labels: $labels,
            metric_type: MetricType::Counter,
        };
    };
    ($name:ident, $description:expr) => {
        pub const $name: MetricInfo = MetricInfo {
            name: stringify!($name),
            description: $description,
            labels: &[],
            metric_type: MetricType::Counter,
        };
    };
}

/// Record a counter metric
#[macro_export]
macro_rules! record_counter {
    ($name:ident, $value:expr) => {
        counter!($name).increment($value);
    };
    ($name:ident, $value:expr, $($label:expr => $val:expr),* $(,)?) => {
        counter!($name $(, $label => $val.to_string())*).increment($value);
    };
}

/// Increment a counter metric
#[macro_export]
macro_rules! increment_counter {
    ($name:ident) => {
        counter!($name).increment(1);
    };
    ($name:ident, $($label:expr => $val:expr),* $(,)?) => {
        counter!($name $(, $label => $val.to_string())*).increment(1);
    };
}

// ============================================================================
// GAUGE MACROS
// ============================================================================

/// Define a gauge metric
#[macro_export]
macro_rules! define_gauge {
    ($name:ident, $description:expr, $labels:expr) => {
        pub const $name: MetricInfo = MetricInfo {
            name: stringify!($name),
            description: $description,
            labels: $labels,
            metric_type: MetricType::Gauge,
        };
    };
    ($name:ident, $description:expr) => {
        pub const $name: MetricInfo = MetricInfo {
            name: stringify!($name),
            description: $description,
            labels: &[],
            metric_type: MetricType::Gauge,
        };
    };
}

/// Record a gauge metric
#[macro_export]
macro_rules! record_gauge {
    ($name:ident, $value:expr) => {
        gauge!($name).set($value);
    };
    ($name:ident, $value:expr, $($label:expr => $val:expr),* $(,)?) => {
        gauge!($name $(, $label => $val.to_string())*).set($value);
    };
}

// ============================================================================
// HISTOGRAM MACROS
// ============================================================================

/// Define a histogram metric
#[macro_export]
macro_rules! define_histogram {
    ($name:ident, $description:expr, $labels:expr) => {
        pub const $name: MetricInfo = MetricInfo {
            name: stringify!($name),
            description: $description,
            labels: $labels,
            metric_type: MetricType::Histogram,
        };
    };
    ($name:ident, $description:expr) => {
        pub const $name: MetricInfo = MetricInfo {
            name: stringify!($name),
            description: $description,
            labels: &[],
            metric_type: MetricType::Histogram,
        };
    };
}

/// Record a histogram metric
#[macro_export]
macro_rules! record_histogram {
    ($name:ident, $value:expr) => {
        histogram!($name).record($value);
    };
    ($name:ident, $value:expr, $($label:expr => $val:expr),* $(,)?) => {
        histogram!($name $(, $label => $val.to_string())*).record($value);
    };
}

// ============================================================================
// METRIC INFO STRUCT
// ============================================================================

/// Metric information structure
#[derive(Debug, Clone)]
pub struct MetricInfo {
    pub name: &'static str,
    pub description: &'static str,
    pub labels: &'static [&'static str],
    pub metric_type: MetricType,
}

/// Metric type enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
}

impl std::fmt::Display for MetricType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetricType::Counter => write!(f, "Counter"),
            MetricType::Gauge => write!(f, "Gauge"),
            MetricType::Histogram => write!(f, "Histogram"),
        }
    }
}

// ============================================================================
// METRIC REGISTRY
// ============================================================================

/// Metric registry for tracking all defined metrics
#[macro_export]
macro_rules! metric_registry {
    ($($name:ident),* $(,)?) => {
        pub fn get_all_metrics() -> Vec<&'static str> {
            vec![$(stringify!($name)),*]
        }
    };
}

// ============================================================================
// INIT METRICS MACRO
// ============================================================================

/// Initialize all metrics with descriptions
#[macro_export]
macro_rules! init_metrics_macro {
    () => {
        /// Initialize all metrics with descriptions
        pub fn init_metrics() {
            // HTTP metrics
            metrics::describe_counter!(
                "http_requests_total",
                "Total HTTP requests by endpoint, method, and status code"
            );
            metrics::describe_histogram!(
                "http_request_duration_seconds",
                "Duration of HTTP requests in seconds"
            );
            metrics::describe_histogram!("http_request_size_bytes", "Size of HTTP requests in bytes");
            metrics::describe_histogram!(
                "http_response_size_bytes",
                "Size of HTTP responses in bytes"
            );

            // Engine metrics
            metrics::describe_counter!("engine_steps_total", "Total engine step operations");
            metrics::describe_histogram!(
                "engine_step_duration_seconds",
                "Duration of engine step operations"
            );
            metrics::describe_histogram!("engine_batch_size", "Batch size per engine step");

            // Inference metrics
            metrics::describe_histogram!(
                "inference_prefill_duration_seconds",
                "Duration of prefill phase"
            );
            metrics::describe_histogram!(
                "inference_decode_duration_seconds",
                "Duration of decode phase"
            );
            metrics::describe_histogram!(
                "inference_prefill_throughput",
                "Tokens per second during prefill"
            );
            metrics::describe_histogram!(
                "inference_decode_throughput",
                "Tokens per second during decode"
            );

            // Sequence metrics
            metrics::describe_gauge!("sequence_count", "Current number of sequences by status");
            metrics::describe_counter!("sequence_added_total", "Total sequences added");
            metrics::describe_counter!("sequence_completed_total", "Total sequences completed");
            metrics::describe_counter!("sequence_cancelled_total", "Total sequences cancelled");

            // KV Cache metrics
            metrics::describe_gauge!("kv_cache_blocks_total", "Total KV cache blocks");
            metrics::describe_gauge!("kv_cache_blocks_used", "Used KV cache blocks");
            metrics::describe_gauge!("kv_cache_usage_percent", "KV cache usage percent");

            // Tool metrics
            metrics::describe_counter!("tool_calls_total", "Total tool calls");
            metrics::describe_histogram!("tool_call_duration_seconds", "Tool call duration");

            // Reasoning metrics
            metrics::describe_counter!("reasoning_blocks_enabled_total", "Reasoning blocks enabled");
            metrics::describe_histogram!("reasoning_tokens_count", "Reasoning token count");

            // Model metrics
            metrics::describe_counter!("model_forward_total", "Model forward passes");
            metrics::describe_histogram!("model_forward_duration_seconds", "Model forward duration");

            // GPU metrics (cuda feature)
            #[cfg(feature = "cuda")]
            {
                metrics::describe_gauge!("gpu_memory_used_bytes", "GPU memory used in bytes");
                metrics::describe_gauge!("gpu_memory_total_bytes", "GPU memory total in bytes");
                metrics::describe_gauge!("gpu_temperature_celsius", "GPU temperature in Celsius");
                metrics::describe_gauge!("gpu_power_watts", "GPU power usage in watts");
            }

            // System metrics
            metrics::describe_gauge!("process_memory_rss_bytes", "Process RSS memory in bytes");
            metrics::describe_gauge!("cpu_utilization_percent", "CPU utilization percent");

            // Scheduler metrics
            metrics::describe_counter!("scheduler_schedule_calls_total", "Total scheduler schedule calls");
            metrics::describe_histogram!("scheduler_schedule_duration_seconds", "Scheduler schedule duration");
            metrics::describe_gauge!("scheduler_queue_waiting_length", "Scheduler waiting queue length");
            metrics::describe_gauge!("scheduler_queue_running_length", "Scheduler running queue length");
            metrics::describe_counter!("scheduler_preemptions_total", "Total scheduler preemptions");
            metrics::describe_counter!("scheduler_swaps_total", "Total scheduler swaps");

            // Block manager metrics
            metrics::describe_counter!("block_manager_allocations_total", "Total block allocations");
            metrics::describe_counter!("block_manager_deallocations_total", "Total block deallocations");
            metrics::describe_counter!("prefix_cache_evictions_total", "Total prefix cache evictions");
            metrics::describe_gauge!("prefix_cache_entries_count", "Prefix cache entries count");

            // Prefix cache metrics
            metrics::describe_counter!("prefix_cache_hits_total", "Total prefix cache hits");
            metrics::describe_counter!("prefix_cache_misses_total", "Total prefix cache misses");
            metrics::describe_histogram!("prefix_cache_hit_ratio", "Prefix cache hit ratio");
            metrics::describe_histogram!("prefix_cache_size_blocks", "Prefix cache size in blocks");

            // Inference metrics
            metrics::describe_counter!("inference_prefill_total", "Total prefill operations");
            metrics::describe_counter!("inference_decode_total", "Total decode operations");
            metrics::describe_histogram!("inference_prefill_tokens", "Prefill tokens per operation");
            metrics::describe_histogram!("inference_decode_tokens", "Decode tokens per operation");

            // Tool call metrics
            metrics::describe_counter!("tool_calls_parsed_total", "Total parsed tool calls");
            metrics::describe_counter!("tool_calls_invalid_total", "Total invalid tool calls");
            metrics::describe_counter!("tool_calls_errors_total", "Total tool call errors");

            // Reasoning metrics
            metrics::describe_counter!("reasoning_blocks_skipped_total", "Reasoning blocks skipped");
            metrics::describe_histogram!("reasoning_duration_seconds", "Reasoning duration");
            metrics::describe_counter!("reasoning_marker_detections_total", "Reasoning marker detections");

            // Model metrics
            metrics::describe_histogram!("model_input_size_tokens", "Model input size in tokens");
            metrics::describe_histogram!("model_output_size_tokens", "Model output size in tokens");

            // CUDA graph metrics
            metrics::describe_counter!("cuda_graph_captures_total", "CUDA graph captures");
            metrics::describe_counter!("cuda_graph_replays_total", "CUDA graph replays");
            metrics::describe_histogram!("cuda_graph_replay_duration_seconds", "CUDA graph replay duration");

            // FlashInfer metrics
            metrics::describe_counter!("flashinfer_attention_total", "FlashInfer attention calls");
            metrics::describe_histogram!("flashinfer_duration_seconds", "FlashInfer duration");

            // IPC metrics
            metrics::describe_counter!("ipc_local_socket_connections_total", "IPC local socket connections");
            metrics::describe_counter!("ipc_local_messages_total", "IPC local messages");
            metrics::describe_counter!("ipc_local_bytes_total", "IPC local bytes");
            metrics::describe_histogram!("ipc_local_transfer_duration_seconds", "IPC local transfer duration");
            metrics::describe_histogram!("ipc_local_transfer_throughput", "IPC local throughput");
            metrics::describe_counter!("ipc_local_errors_total", "IPC local errors");
            metrics::describe_counter!("ipc_tcp_socket_connections_total", "IPC TCP socket connections");
            metrics::describe_counter!("ipc_tcp_messages_total", "IPC TCP messages");
            metrics::describe_counter!("ipc_tcp_bytes_total", "IPC TCP bytes");
            metrics::describe_histogram!("ipc_tcp_transfer_duration_seconds", "IPC TCP transfer duration");
            metrics::describe_histogram!("ipc_tcp_transfer_throughput", "IPC TCP throughput");
            metrics::describe_histogram!("ipc_tcp_transfer_latency_seconds", "IPC TCP latency");
            metrics::describe_counter!("ipc_tcp_errors_total", "IPC TCP errors");
            metrics::describe_counter!("ipc_tcp_retransmits_total", "IPC TCP retransmits");
            metrics::describe_counter!("ipc_tcp_timeouts_total", "IPC TCP timeouts");

            // PD metrics
            metrics::describe_counter!("pd_prefill_sent_total", "PD prefill sent");
            metrics::describe_counter!("pd_prefill_received_total", "PD prefill received");
            metrics::describe_histogram!("pd_prefill_transfer_duration_seconds", "PD prefill transfer duration");
            metrics::describe_histogram!("pd_prefill_transfer_size_bytes", "PD prefill transfer size");
            metrics::describe_counter!("pd_prefill_transfer_failures_total", "PD prefill transfer failures");

            // GPU metrics
            metrics::describe_gauge!("gpu_memory_free_bytes", "GPU memory free in bytes");
            metrics::describe_gauge!("gpu_memory_utilization_percent", "GPU memory utilization percent");
            metrics::describe_gauge!("gpu_sm_utilization_percent", "GPU SM utilization percent");
            metrics::describe_gauge!("gpu_clock_sm_mhz", "GPU SM clock in MHz");
            metrics::describe_gauge!("gpu_clock_memory_mhz", "GPU memory clock in MHz");
            metrics::describe_counter!("gpu_errors_correctable_total", "GPU correctable errors");
            metrics::describe_counter!("gpu_errors_non_correctable_total", "GPU non-correctable errors");

            // Process metrics
            metrics::describe_gauge!("process_memory_vms_bytes", "Process virtual memory in bytes");
            metrics::describe_gauge!("process_memory_heap_bytes", "Process heap memory in bytes");
            metrics::describe_gauge!("process_memory_stack_bytes", "Process stack memory in bytes");
            metrics::describe_gauge!("system_memory_used_bytes", "System memory used in bytes");
            metrics::describe_gauge!("system_memory_total_bytes", "System memory total in bytes");
            metrics::describe_gauge!("system_load_avg1", "System load average (1 minute)");
            metrics::describe_gauge!("system_load_avg5", "System load average (5 minutes)");
            metrics::describe_gauge!("system_load_avg15", "System load average (15 minutes)");
            metrics::describe_gauge!("process_thread_count", "Process thread count");
            metrics::describe_gauge!("runtime_active_tasks_count", "Runtime active tasks count");
            metrics::describe_gauge!("runtime_blocked_threads_count", "Runtime blocked threads count");
            metrics::describe_gauge!("runtime_num_workers_count", "Runtime worker count");

            // Event loop metrics
            metrics::describe_histogram!("runtime_event_loop_delay_seconds", "Event loop delay");
            metrics::describe_histogram!("runtime_blocking_time_seconds", "Blocking time");

            // Mamba metrics
            metrics::describe_counter!("mamba_snapshot_captures_total", "Mamba snapshot captures");
            metrics::describe_counter!("mamba_snapshot_misses_total", "Mamba snapshot misses");
            metrics::describe_histogram!("mamba_restore_duration_seconds", "Mamba restore duration");

            // Throughput metrics
            metrics::describe_histogram!("tokens_per_second", "Token generation rate");
            metrics::describe_histogram!("request_queue_wait_seconds", "Request queue wait time");
            metrics::describe_histogram!("request_latency_seconds", "Request latency");
            metrics::describe_histogram!("request_prompt_tokens_count", "Prompt tokens per request");
            metrics::describe_histogram!("request_completion_tokens_count", "Completion tokens per request");

            // Error metrics
            metrics::describe_counter!("errors_total", "Total errors by category");
            metrics::describe_counter!("validation_errors_total", "Validation errors by endpoint");
            metrics::describe_counter!("model_execution_errors_total", "Model execution errors");

            // Token metrics
            metrics::describe_counter!("tokens_prompt_total", "Total prompt tokens processed");
            metrics::describe_counter!("tokens_completion_total", "Total completion tokens generated");
            metrics::describe_histogram!("tokens_per_second_prompt", "Prompt tokens per second");
            metrics::describe_histogram!("tokens_per_second_decode", "Decode tokens per second");
            metrics::describe_histogram!("tokens_per_request_count", "Tokens per request");

            // Worker metrics
            metrics::describe_gauge!("worker_utilization_percent", "Worker utilization percent");

            // Cache metrics
            metrics::describe_counter!("tokenizer_cache_hits_total", "Tokenizer cache hits");
            metrics::describe_counter!("tokenizer_cache_misses_total", "Tokenizer cache misses");
            metrics::describe_histogram!("tokenizer_cache_duration_seconds", "Tokenizer cache duration");
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_info() {
        let info = MetricInfo {
            name: "test_metric",
            description: "Test metric description",
            labels: &["label1", "label2"],
            metric_type: MetricType::Counter,
        };
        assert_eq!(info.name, "test_metric");
        assert_eq!(info.description, "Test metric description");
        assert_eq!(info.labels.len(), 2);
        assert_eq!(info.metric_type, MetricType::Counter);
    }

    #[test]
    fn test_metric_type() {
        assert_eq!(MetricType::Counter.to_string(), "Counter");
        assert_eq!(MetricType::Gauge.to_string(), "Gauge");
        assert_eq!(MetricType::Histogram.to_string(), "Histogram");
    }
}