// src/utils/metrics/metrics.rs
//! Core metrics recording module for vllm.rs instrumentation
//!
//! Provides all metric recording functions using the metrics crate.
//! This module is the single source of truth for all metrics in the system.
//!
//! # Design Principles
//!
//! - **DRY**: One function per metric type, no duplication
//! - **Explicit**: All metric names and labels are explicit
//! - **Production-grade**: Proper error handling and testing
//!
//! # Usage
//!
//! ```rust
//! use xinfer::utils::metrics;
//!
//! // Record a counter
//! metrics::record_http_request_total("/v1/chat/completions", "POST", "200");
//!
//! // Record a histogram
//! metrics::record_engine_step_duration(0.123);
//!
//! // Record a gauge
//! metrics::record_gpu_temperature("0", 75.5);
//! ```

use metrics::{counter, gauge, histogram};

// ============================================================================
// HTTP REQUEST METRICS
// ============================================================================

/// Record HTTP request total count
pub fn record_http_request_total(endpoint: &str, method: &str, status: &str) {
    counter!("http_requests_total",
        "endpoint" => endpoint.to_string(),
        "method" => method.to_string(),
        "status" => status.to_string()
    )
    .increment(1);
}

/// Record HTTP request duration histogram
pub fn record_http_request_duration(endpoint: &str, method: &str, duration_seconds: f64) {
    histogram!("http_request_duration_seconds",
        "endpoint" => endpoint.to_string(),
        "method" => method.to_string()
    )
    .record(duration_seconds);
}

/// Record HTTP request size
pub fn record_http_request_size(endpoint: &str, method: &str, size_bytes: u64) {
    histogram!("http_request_size_bytes",
        "endpoint" => endpoint.to_string(),
        "method" => method.to_string()
    )
    .record(size_bytes as f64);
}

/// Record HTTP response size
pub fn record_http_response_size(endpoint: &str, method: &str, size_bytes: u64) {
    histogram!("http_response_size_bytes",
        "endpoint" => endpoint.to_string(),
        "method" => method.to_string()
    )
    .record(size_bytes as f64);
}

// ============================================================================
// ENGINE METRICS
// ============================================================================

/// Record engine step total
pub fn record_engine_step_total() {
    counter!("engine_steps_total").increment(1);
}

/// Record engine step duration
pub fn record_engine_step_duration(duration_seconds: f64) {
    histogram!("engine_step_duration_seconds").record(duration_seconds);
}

/// Record engine batch size
pub fn record_engine_batch_size(batch_size: usize) {
    histogram!("engine_batch_size").record(batch_size as f64);
}

// ============================================================================
// ENGINE PHASE METRICS
// ============================================================================

/// Record prepare_step duration
pub fn record_prepare_step_duration(duration_seconds: f64) {
    histogram!("engine_prepare_step_duration_seconds").record(duration_seconds);
}

/// Record run_forward duration
pub fn record_run_forward_duration(duration_seconds: f64) {
    histogram!("engine_run_forward_duration_seconds").record(duration_seconds);
}

/// Record finish_step duration
pub fn record_finish_step_duration(duration_seconds: f64) {
    histogram!("engine_finish_step_duration_seconds").record(duration_seconds);
}

// ============================================================================
// PREFILL/DECODE METRICS
// ============================================================================

/// Record prefill total
pub fn record_prefill_total(sequence_id: u64) {
    counter!("inference_prefill_total", "sequence_id" => sequence_id.to_string()).increment(1);
}

/// Record prefill duration
pub fn record_prefill_duration(sequence_id: u64, duration_seconds: f64) {
    histogram!("inference_prefill_duration_seconds", "sequence_id" => sequence_id.to_string()).record(duration_seconds);
}

/// Record prefill tokens
pub fn record_prefill_tokens(sequence_id: u64, token_count: u64) {
    histogram!("inference_prefill_tokens", "sequence_id" => sequence_id.to_string()).record(token_count as f64);
}

/// Record prefill throughput (tokens/second)
pub fn record_prefill_throughput(sequence_id: u64, tokens_per_second: f64) {
    histogram!("inference_prefill_throughput", "sequence_id" => sequence_id.to_string()).record(tokens_per_second);
}

/// Record decode total
pub fn record_decode_total(sequence_id: u64) {
    counter!("inference_decode_total", "sequence_id" => sequence_id.to_string()).increment(1);
}

/// Record decode duration
pub fn record_decode_duration(sequence_id: u64, duration_seconds: f64) {
    histogram!("inference_decode_duration_seconds", "sequence_id" => sequence_id.to_string()).record(duration_seconds);
}

/// Record decode tokens
pub fn record_decode_tokens(sequence_id: u64, token_count: u64) {
    histogram!("inference_decode_tokens", "sequence_id" => sequence_id.to_string()).record(token_count as f64);
}

/// Record decode throughput (tokens/second)
pub fn record_decode_throughput(sequence_id: u64, tokens_per_second: f64) {
    histogram!("inference_decode_throughput", "sequence_id" => sequence_id.to_string()).record(tokens_per_second);
}

// ============================================================================
// SEQUENCE METRICS
// ============================================================================

/// Record sequence count by status
pub fn record_sequence_count(status: &str, count: u64) {
    gauge!("sequence_count", "status" => status.to_string()).set(count as f64);
}

/// Record sequence added
pub fn record_sequence_added(sequence_id: u64) {
    counter!("sequence_added_total", "sequence_id" => sequence_id.to_string()).increment(1);
}

/// Record sequence completed
pub fn record_sequence_completed(sequence_id: u64, finish_reason: &str) {
    counter!("sequence_completed_total",
        "sequence_id" => sequence_id.to_string(),
        "finish_reason" => finish_reason.to_string()
    )
    .increment(1);
}

/// Record sequence cancelled
pub fn record_sequence_cancelled(sequence_id: u64, reason: &str) {
    counter!("sequence_cancelled_total",
        "sequence_id" => sequence_id.to_string(),
        "reason" => reason.to_string()
    )
    .increment(1);
}

/// Record sequence wait time
pub fn record_sequence_wait_time(duration_seconds: f64) {
    histogram!("sequence_wait_time_seconds").record(duration_seconds);
}

/// Record sequence total time
pub fn record_sequence_total_time(duration_seconds: f64) {
    histogram!("sequence_total_time_seconds").record(duration_seconds);
}

// ============================================================================
// TTFT (TIME TO FIRST TOKEN) METRICS
// ============================================================================

/// Record TTFT for a sequence
pub fn record_ttft(sequence_id: u64, duration_seconds: f64) {
    histogram!("ttft_seconds", "sequence_id" => sequence_id.to_string()).record(duration_seconds);
}

/// Record TTFT percentile (p50, p99, p999)
pub fn record_ttft_percentile(percentile: &str, duration_seconds: f64) {
    histogram!("ttft_percentile_seconds", "percentile" => percentile.to_string()).record(duration_seconds);
}

// ============================================================================
// KV CACHE METRICS
// ============================================================================

/// Record KV cache blocks total
pub fn record_kv_cache_blocks_total(device: &str, count: u64) {
    gauge!("kv_cache_blocks_total", "device" => device.to_string()).set(count as f64);
}

/// Record KV cache blocks used
pub fn record_kv_cache_blocks_used(device: &str, count: u64) {
    gauge!("kv_cache_blocks_used", "device" => device.to_string()).set(count as f64);
}

/// Record KV cache blocks free
pub fn record_kv_cache_blocks_free(device: &str, count: u64) {
    gauge!("kv_cache_blocks_free", "device" => device.to_string()).set(count as f64);
}

/// Record KV cache usage percent
pub fn record_kv_cache_usage_percent(device: &str, percent: f64) {
    gauge!("kv_cache_usage_percent", "device" => device.to_string()).set(percent);
}

/// Record KV cache evictions
pub fn record_kv_cache_evictions(device: &str, count: u64) {
    counter!("kv_cache_evictions_total", "device" => device.to_string()).increment(count);
}

/// Record KV cache swap out
pub fn record_kv_cache_swap_out(device: &str, count: u64) {
    counter!("kv_cache_swap_out_total", "device" => device.to_string()).increment(count);
}

/// Record KV cache swap in
pub fn record_kv_cache_swap_in(device: &str, count: u64) {
    counter!("kv_cache_swap_in_total", "device" => device.to_string()).increment(count);
}

// ============================================================================
// PREFIX CACHE METRICS
// ============================================================================

/// Record prefix cache hits
pub fn record_prefix_cache_hit(sequence_id: u64) {
    counter!("prefix_cache_hits_total", "sequence_id" => sequence_id.to_string()).increment(1);
}

/// Record prefix cache misses
pub fn record_prefix_cache_miss(sequence_id: u64) {
    counter!("prefix_cache_misses_total", "sequence_id" => sequence_id.to_string()).increment(1);
}

/// Record prefix cache hit ratio
pub fn record_prefix_cache_hit_ratio(ratio: f64) {
    histogram!("prefix_cache_hit_ratio").record(ratio);
}

/// Record prefix cache size in blocks
pub fn record_prefix_cache_size(blocks: u64) {
    histogram!("prefix_cache_size_blocks").record(blocks as f64);
}

// ============================================================================
// GRAMMAR/GUIDANCE METRICS
// ============================================================================

/// Record grammar compilations
pub fn record_grammar_compile(sequence_id: u64, grammar_type: &str) {
    counter!("grammar_compiles_total", "sequence_id" => sequence_id.to_string(), "type" => grammar_type.to_string()).increment(1);
}

/// Record grammar compilation duration
pub fn record_grammar_compile_duration(sequence_id: u64, grammar_type: &str, duration_seconds: f64) {
    histogram!("grammar_compile_duration_seconds", "sequence_id" => sequence_id.to_string(), "type" => grammar_type.to_string())
        .record(duration_seconds);
}

/// Record guidance mask computations
pub fn record_guidance_mask_computation(sequence_id: u64) {
    counter!("guidance_masks_total", "sequence_id" => sequence_id.to_string()).increment(1);
}

/// Record guidance mask duration
pub fn record_guidance_mask_duration(sequence_id: u64, duration_seconds: f64) {
    histogram!("guidance_mask_duration_seconds", "sequence_id" => sequence_id.to_string()).record(duration_seconds);
}

/// Record constrained token generations
pub fn record_constrained_token_generation(sequence_id: u64) {
    counter!("constrained_tokens_total", "sequence_id" => sequence_id.to_string()).increment(1);
}

// ============================================================================
// TOOL CALLING METRICS
// ============================================================================

/// Record tool calls
pub fn record_tool_call(tool_name: &str, sequence_id: u64) {
    counter!("tool_calls_total",
        "tool" => tool_name.to_string(),
        "sequence_id" => sequence_id.to_string()
    )
    .increment(1);
}

/// Record tool call duration
pub fn record_tool_call_duration(tool_name: &str, sequence_id: u64, duration_seconds: f64) {
    histogram!("tool_call_duration_seconds",
        "tool" => tool_name.to_string(),
        "sequence_id" => sequence_id.to_string()
    )
    .record(duration_seconds);
}

/// Record tool call errors
pub fn record_tool_call_error(tool_name: &str, sequence_id: u64, error_type: &str) {
    counter!("tool_calls_errors_total",
        "tool" => tool_name.to_string(),
        "sequence_id" => sequence_id.to_string(),
        "error_type" => error_type.to_string()
    )
    .increment(1);
}

/// Record tool call parsed
pub fn record_tool_call_parsed(tool_name: &str, sequence_id: u64) {
    counter!("tool_calls_parsed_total", "tool" => tool_name.to_string(), "sequence_id" => sequence_id.to_string()).increment(1);
}

/// Record tool call invalid
pub fn record_tool_call_invalid(tool_name: &str, sequence_id: u64) {
    counter!("tool_calls_invalid_total", "tool" => tool_name.to_string(), "sequence_id" => sequence_id.to_string()).increment(1);
}

// ============================================================================
// REASONING METRICS
// ============================================================================

/// Record reasoning blocks enabled
pub fn record_reasoning_enabled(sequence_id: u64) {
    counter!("reasoning_blocks_enabled_total", "sequence_id" => sequence_id.to_string()).increment(1);
}

/// Record reasoning blocks skipped
pub fn record_reasoning_skipped(sequence_id: u64, reason: &str) {
    counter!("reasoning_blocks_skipped_total",
        "sequence_id" => sequence_id.to_string(),
        "reason" => reason.to_string()
    )
    .increment(1);
}

/// Record reasoning tokens
pub fn record_reasoning_tokens(sequence_id: u64, token_count: u64) {
    histogram!("reasoning_tokens_count", "sequence_id" => sequence_id.to_string()).record(token_count as f64);
}

/// Record reasoning duration
pub fn record_reasoning_duration(sequence_id: u64, duration_seconds: f64) {
    histogram!("reasoning_duration_seconds", "sequence_id" => sequence_id.to_string()).record(duration_seconds);
}

/// Record reasoning marker detection
pub fn record_reasoning_marker_detection(sequence_id: u64, marker: &str) {
    counter!("reasoning_marker_detections_total", "sequence_id" => sequence_id.to_string(), "marker" => marker.to_string()).increment(1);
}

// ============================================================================
// RUNNER MODEL EXECUTION METRICS
// ============================================================================

/// Record model forward pass
pub fn record_model_forward(sequence_id: u64, model_type: &str, phase: &str) {
    counter!("model_forward_total",
        "sequence_id" => sequence_id.to_string(),
        "model_type" => model_type.to_string(),
        "phase" => phase.to_string()
    )
    .increment(1);
}

/// Record model forward duration
pub fn record_model_forward_duration(sequence_id: u64, model_type: &str, phase: &str, duration_seconds: f64) {
    histogram!("model_forward_duration_seconds",
        "sequence_id" => sequence_id.to_string(),
        "model_type" => model_type.to_string(),
        "phase" => phase.to_string()
    )
    .record(duration_seconds);
}

/// Record model input size
pub fn record_model_input_size(sequence_id: u64, model_type: &str, phase: &str, token_count: u64) {
    histogram!("model_input_size_tokens",
        "sequence_id" => sequence_id.to_string(),
        "model_type" => model_type.to_string(),
        "phase" => phase.to_string()
    )
    .record(token_count as f64);
}

/// Record model output size
pub fn record_model_output_size(sequence_id: u64, model_type: &str, phase: &str, token_count: u64) {
    histogram!("model_output_size_tokens",
        "sequence_id" => sequence_id.to_string(),
        "model_type" => model_type.to_string(),
        "phase" => phase.to_string()
    )
    .record(token_count as f64);
}

// ============================================================================
// CUDA GRAPH METRICS
// ============================================================================

/// Record CUDA graph captures
pub fn record_cuda_graph_capture(sequence_id: u64, model_type: &str) {
    counter!("cuda_graph_captures_total", "sequence_id" => sequence_id.to_string(), "model_type" => model_type.to_string()).increment(1);
}

/// Record CUDA graph replays
pub fn record_cuda_graph_replay(sequence_id: u64, model_type: &str) {
    counter!("cuda_graph_replays_total", "sequence_id" => sequence_id.to_string(), "model_type" => model_type.to_string()).increment(1);
}

/// Record CUDA graph replay duration
pub fn record_cuda_graph_replay_duration(sequence_id: u64, model_type: &str, duration_seconds: f64) {
    histogram!("cuda_graph_replay_duration_seconds", "sequence_id" => sequence_id.to_string(), "model_type" => model_type.to_string())
        .record(duration_seconds);
}

// ============================================================================
// FLASHINFER METRICS
// ============================================================================

/// Record flashinfer attention
pub fn record_flashinfer_attention(phase: &str) {
    counter!("flashinfer_attention_total", "phase" => phase.to_string()).increment(1);
}

/// Record flashinfer duration
pub fn record_flashinfer_duration(phase: &str, duration_seconds: f64) {
    histogram!("flashinfer_duration_seconds", "phase" => phase.to_string())
        .record(duration_seconds);
}

// ============================================================================
// BATCH SCHEDULER METRICS
// ============================================================================

/// Record scheduler schedule calls
pub fn record_scheduler_schedule_call() {
    counter!("scheduler_schedule_calls_total").increment(1);
}

/// Record scheduler schedule duration
pub fn record_scheduler_schedule_duration(duration_seconds: f64) {
    histogram!("scheduler_schedule_duration_seconds").record(duration_seconds);
}

/// Record scheduler waiting queue length
pub fn record_scheduler_waiting_length(length: u64) {
    gauge!("scheduler_queue_waiting_length").set(length as f64);
}

/// Record scheduler running queue length
pub fn record_scheduler_running_length(length: u64) {
    gauge!("scheduler_queue_running_length").set(length as f64);
}

/// Record scheduler preemptions
pub fn record_scheduler_preemption() {
    counter!("scheduler_preemptions_total").increment(1);
}

/// Record scheduler swaps
pub fn record_scheduler_swap(direction: &str) {
    counter!("scheduler_swaps_total", "direction" => direction.to_string()).increment(1);
}

// ============================================================================
// IPC/TRANSFER METRICS
// ============================================================================

// LocalIpc Metrics
/// Record IPC local socket connections
pub fn record_ipc_local_socket_connections() {
    counter!("ipc_local_socket_connections_total").increment(1);
}

/// Record IPC local messages
pub fn record_ipc_local_messages(direction: &str) {
    counter!("ipc_local_messages_total", "direction" => direction.to_string()).increment(1);
}

/// Record IPC local bytes
pub fn record_ipc_local_bytes(direction: &str, bytes: u64) {
    counter!("ipc_local_bytes_total", "direction" => direction.to_string()).increment(bytes);
}

/// Record IPC local transfer duration
pub fn record_ipc_local_transfer_duration(duration_seconds: f64) {
    histogram!("ipc_local_transfer_duration_seconds").record(duration_seconds);
}

/// Record IPC local throughput
pub fn record_ipc_local_throughput(mb_per_second: f64) {
    histogram!("ipc_local_transfer_throughput").record(mb_per_second);
}

/// Record IPC local errors
pub fn record_ipc_local_error(error_type: &str) {
    counter!("ipc_local_errors_total", "error_type" => error_type.to_string()).increment(1);
}

// RemoteTcp Metrics
/// Record IPC TCP socket connections
pub fn record_ipc_tcp_socket_connections() {
    counter!("ipc_tcp_socket_connections_total").increment(1);
}

/// Record IPC TCP messages
pub fn record_ipc_tcp_messages(direction: &str) {
    counter!("ipc_tcp_messages_total", "direction" => direction.to_string()).increment(1);
}

/// Record IPC TCP bytes
pub fn record_ipc_tcp_bytes(direction: &str, bytes: u64) {
    counter!("ipc_tcp_bytes_total", "direction" => direction.to_string()).increment(bytes);
}

/// Record IPC TCP transfer duration
pub fn record_ipc_tcp_transfer_duration(duration_seconds: f64) {
    histogram!("ipc_tcp_transfer_duration_seconds").record(duration_seconds);
}

/// Record IPC TCP throughput
pub fn record_ipc_tcp_throughput(mb_per_second: f64) {
    histogram!("ipc_tcp_transfer_throughput").record(mb_per_second);
}

/// Record IPC TCP latency
pub fn record_ipc_tcp_latency(duration_seconds: f64) {
    histogram!("ipc_tcp_transfer_latency_seconds").record(duration_seconds);
}

/// Record IPC TCP errors
pub fn record_ipc_tcp_error(error_type: &str) {
    counter!("ipc_tcp_errors_total", "error_type" => error_type.to_string()).increment(1);
}

/// Record IPC TCP retransmits
pub fn record_ipc_tcp_retransmits() {
    counter!("ipc_tcp_retransmits_total").increment(1);
}

/// Record IPC TCP timeouts
pub fn record_ipc_tcp_timeouts() {
    counter!("ipc_tcp_timeouts_total").increment(1);
}

// PD Prefill Transfer Metrics
/// Record PD prefill sent
pub fn record_pd_prefill_sent() {
    counter!("pd_prefill_sent_total").increment(1);
}

/// Record PD prefill received
pub fn record_pd_prefill_received() {
    counter!("pd_prefill_received_total").increment(1);
}

/// Record PD prefill transfer duration
pub fn record_pd_prefill_transfer_duration(duration_seconds: f64) {
    histogram!("pd_prefill_transfer_duration_seconds").record(duration_seconds);
}

/// Record PD prefill transfer size
pub fn record_pd_prefill_transfer_size(bytes: u64) {
    histogram!("pd_prefill_transfer_size_bytes").record(bytes as f64);
}

/// Record PD prefill transfer failure
pub fn record_pd_prefill_transfer_failure() {
    counter!("pd_prefill_transfer_failures_total").increment(1);
}

/// Record PD prefill throughput - GAP FILL
pub fn record_pd_prefill_throughput(tokens_per_second: f64) {
    histogram!("pd_prefill_throughput_tokens_per_second").record(tokens_per_second);
}

/// Record KV cache transfer duration - GAP FILL
pub fn record_kvcache_transfer_duration(duration_seconds: f64) {
    histogram!("kvcache_transfer_duration_seconds").record(duration_seconds);
}
#[cfg(feature = "cuda")]
/// Record GPU metrics collection errors - GAP FILL
pub fn record_gpu_metrics_errors(error_type: &str) {
    counter!("gpu_metrics_errors_total", "error_type" => error_type.to_string()).increment(1);
}
#[cfg(feature = "cuda")]
/// Record GPU metrics collection errors with source identifier
pub fn record_gpu_metrics_errors_with_source(source: &str, error_type: &str) {
    counter!("gpu_metrics_errors_total", "source" => source.to_string(), "error_type" => error_type.to_string()).increment(1);
}

// ============================================================================
// GPU METRICS (cuda feature gated)
// ============================================================================

/// Record GPU memory used
#[cfg(feature = "cuda")]
pub fn record_gpu_memory_used(device_id: &str, bytes: u64) {
    gauge!("gpu_memory_used_bytes", "device_id" => device_id.to_string()).set(bytes as f64);
}

/// Record GPU memory total
#[cfg(feature = "cuda")]
pub fn record_gpu_memory_total(device_id: &str, bytes: u64) {
    gauge!("gpu_memory_total_bytes", "device_id" => device_id.to_string()).set(bytes as f64);
}

/// Record GPU memory free
#[cfg(feature = "cuda")]
pub fn record_gpu_memory_free(device_id: &str, bytes: u64) {
    gauge!("gpu_memory_free_bytes", "device_id" => device_id.to_string()).set(bytes as f64);
}

/// Record GPU memory utilization percent
#[cfg(feature = "cuda")]
pub fn record_gpu_memory_utilization(device_id: &str, percent: f64) {
    gauge!("gpu_memory_utilization_percent", "device_id" => device_id.to_string()).set(percent);
}

/// Record GPU SM utilization percent
#[cfg(feature = "cuda")]
pub fn record_gpu_sm_utilization(device_id: &str, percent: f64) {
    gauge!("gpu_sm_utilization_percent", "device_id" => device_id.to_string()).set(percent);
}

/// Record GPU temperature in Celsius
#[cfg(feature = "cuda")]
pub fn record_gpu_temperature(device_id: &str, celsius: f64) {
    gauge!("gpu_temperature_celsius", "device_id" => device_id.to_string()).set(celsius);
}

/// Record GPU power usage in watts
#[cfg(feature = "cuda")]
pub fn record_gpu_power_watts(device_id: &str, watts: f64) {
    gauge!("gpu_power_watts", "device_id" => device_id.to_string()).set(watts);
}

/// Record GPU SM clock in MHz
#[cfg(feature = "cuda")]
pub fn record_gpu_clock_sm(device_id: &str, mhz: u64) {
    gauge!("gpu_clock_sm_mhz", "device_id" => device_id.to_string()).set(mhz as f64);
}

/// Record GPU memory clock in MHz
#[cfg(feature = "cuda")]
pub fn record_gpu_clock_memory(device_id: &str, mhz: u64) {
    gauge!("gpu_clock_memory_mhz", "device_id" => device_id.to_string()).set(mhz as f64);
}

/// Record GPU correctable errors
#[cfg(feature = "cuda")]
pub fn record_gpu_correctable_errors(device_id: &str, count: u64) {
    counter!("gpu_errors_correctable_total", "device_id" => device_id.to_string()).increment(count);
}

/// Record GPU non-correctable errors
#[cfg(feature = "cuda")]
pub fn record_gpu_non_correctable_errors(device_id: &str, count: u64) {
    counter!("gpu_errors_non_correctable_total", "device_id" => device_id.to_string())
        .increment(count);
}

// ============================================================================
// CPU/PROCESS METRICS
// ============================================================================

/// Record process RSS memory in bytes
pub fn record_process_memory_rss(bytes: u64) {
    gauge!("process_memory_rss_bytes").set(bytes as f64);
}

/// Record process virtual memory in bytes
pub fn record_process_memory_vms(bytes: u64) {
    gauge!("process_memory_vms_bytes").set(bytes as f64);
}

/// Record process heap memory in bytes
pub fn record_process_memory_heap(bytes: u64) {
    gauge!("process_memory_heap_bytes").set(bytes as f64);
}

/// Record process stack memory in bytes
pub fn record_process_memory_stack(bytes: u64) {
    gauge!("process_memory_stack_bytes").set(bytes as f64);
}

/// Record CPU utilization percent
pub fn record_cpu_utilization(percent: f64) {
    gauge!("cpu_utilization_percent").set(percent);
}

/// Record system memory used in bytes
pub fn record_system_memory_used(bytes: u64) {
    gauge!("system_memory_used_bytes").set(bytes as f64);
}

/// Record system memory total in bytes
pub fn record_system_memory_total(bytes: u64) {
    gauge!("system_memory_total_bytes").set(bytes as f64);
}

/// Record load average (1 minute)
pub fn record_load_avg_1(minute: f64) {
    gauge!("system_load_avg1").set(minute);
}

/// Record load average (5 minutes)
pub fn record_load_avg_5(minute: f64) {
    gauge!("system_load_avg5").set(minute);
}

/// Record load average (15 minutes)
pub fn record_load_avg_15(minute: f64) {
    gauge!("system_load_avg15").set(minute);
}

/// Record thread count
pub fn record_thread_count(count: u64) {
    gauge!("process_thread_count").set(count as f64);
}

/// Record active task count
pub fn record_active_tasks(count: u64) {
    gauge!("runtime_active_tasks_count").set(count as f64);
}

// ============================================================================
// EVENT LOOP METRICS
// ============================================================================

/// Record event loop delay in seconds
pub fn record_event_loop_delay(duration_seconds: f64) {
    histogram!("runtime_event_loop_delay_seconds").record(duration_seconds);
}

/// Record runtime blocking time in seconds
pub fn record_runtime_blocking_time(duration_seconds: f64) {
    histogram!("runtime_blocking_time_seconds").record(duration_seconds);
}

// ============================================================================
// MAMBA METRICS
// ============================================================================

/// Record mamba snapshot capture
pub fn record_mamba_snapshot_capture(sequence_id: u64) {
    counter!("mamba_snapshot_captures_total", "sequence_id" => sequence_id.to_string()).increment(1);
}

/// Record mamba snapshot miss
pub fn record_mamba_snapshot_miss(sequence_id: u64) {
    counter!("mamba_snapshot_misses_total", "sequence_id" => sequence_id.to_string()).increment(1);
}

/// Record mamba restore duration
pub fn record_mamba_restore_duration(sequence_id: u64, duration_seconds: f64) {
    histogram!("mamba_restore_duration_seconds", "sequence_id" => sequence_id.to_string()).record(duration_seconds);
}

/// Record mamba slot usage percent - GAP FILL
pub fn record_mamba_slot_usage_percent(sequence_id: u64, percent: f64) {
    gauge!("mamba_slot_usage_percent", "sequence_id" => sequence_id.to_string()).set(percent);
}

/// Record mamba memory used bytes - GAP FILL
pub fn record_mamba_memory_used_bytes(sequence_id: u64, bytes: u64) {
    gauge!("mamba_memory_used_bytes", "sequence_id" => sequence_id.to_string()).set(bytes as f64);
}

// ============================================================================
// BLOCK MANAGER METRICS
// ============================================================================

/// Record block allocation
pub fn record_block_allocation(sequence_id: u64) {
    counter!("block_manager_allocations_total", "sequence_id" => sequence_id.to_string()).increment(1);
}

/// Record block deallocation
pub fn record_block_deallocation(sequence_id: u64) {
    counter!("block_manager_deallocations_total", "sequence_id" => sequence_id.to_string()).increment(1);
}

/// Record prefix cache eviction
pub fn record_prefix_cache_eviction(sequence_id: u64) {
    counter!("prefix_cache_evictions_total", "sequence_id" => sequence_id.to_string()).increment(1);
}

/// Record prefix cache entries
pub fn record_prefix_cache_entries(sequence_id: u64, count: u64) {
    gauge!("prefix_cache_entries_count", "sequence_id" => sequence_id.to_string()).set(count as f64);
}

/// Record block allocation (system-level, no sequence_id)
pub fn record_block_allocation_system() {
    counter!("block_manager_allocations_total_system").increment(1);
}

/// Record block deallocation (system-level, no sequence_id)
/// Also records to block_manager_deallocations_total for backward compatibility
pub fn record_block_deallocation_system() {
    counter!("block_manager_deallocations_total_system").increment(1);
    counter!("block_manager_deallocations_total").increment(1);
}

/// Record prefix cache eviction (system-level, no sequence_id)
pub fn record_prefix_cache_eviction_system() {
    counter!("prefix_cache_evictions_total_system").increment(1);
}

/// Record prefix cache entries (system-level, no sequence_id)
pub fn record_prefix_cache_entries_system(count: u64) {
    gauge!("prefix_cache_entries_count_system").set(count as f64);
}

// ============================================================================
// UTILIZATION GAUGES
// ============================================================================

/// Update all utilization gauges at once
pub fn update_utilization_gauges(
    device_id: &str,
    gpu_util: f64,
    gpu_mem_util: f64,
    cpu_util: f64,
    system_mem_util: f64,
) {
    gauge!("gpu_utilization_percent", "device_id" => device_id.to_string()).set(gpu_util);
    gauge!("gpu_memory_utilization_percent", "device_id" => device_id.to_string())
        .set(gpu_mem_util);
    gauge!("cpu_utilization_percent").set(cpu_util);
    gauge!("system_memory_utilization_percent").set(system_mem_util);
}

// ============================================================================
// SAMPLING METRICS
// ============================================================================

/// Record token generation rate
pub fn record_token_generation_rate(sequence_id: u64, rate: f64) {
    histogram!("tokens_per_second", "sequence_id" => sequence_id.to_string()).record(rate);
}

/// Record request queue wait time
pub fn record_request_queue_wait(sequence_id: u64, duration_seconds: f64) {
    histogram!("request_queue_wait_seconds", "sequence_id" => sequence_id.to_string()).record(duration_seconds);
}

/// Record end-to-end request latency
pub fn record_request_latency(sequence_id: u64, duration_seconds: f64) {
    histogram!("request_latency_seconds", "sequence_id" => sequence_id.to_string()).record(duration_seconds);
}

/// Record prompt tokens per request
pub fn record_prompt_tokens_per_request(sequence_id: u64, count: u64) {
    histogram!("request_prompt_tokens_count", "sequence_id" => sequence_id.to_string()).record(count as f64);
}

/// Record completion tokens per request
pub fn record_completion_tokens_per_request(sequence_id: u64, count: u64) {
    histogram!("request_completion_tokens_count", "sequence_id" => sequence_id.to_string()).record(count as f64);
}

// ============================================================================
// ERROR METRICS
// ============================================================================

/// Record error by category
pub fn record_error(category: &str, error_type: &str) {
    counter!("errors_total",
        "category" => category.to_string(),
        "error_type" => error_type.to_string()
    )
    .increment(1);
}

/// Record validation error
pub fn record_validation_error(endpoint: &str, error_type: &str) {
    counter!("validation_errors_total",
        "endpoint" => endpoint.to_string(),
        "error_type" => error_type.to_string()
    )
    .increment(1);
}

/// Record model execution error
pub fn record_model_execution_error(sequence_id: u64, error_type: &str) {
    counter!("model_execution_errors_total",
        "sequence_id" => sequence_id.to_string(),
        "error_type" => error_type.to_string()
    )
    .increment(1);
}

// ============================================================================
// THROUGHPUT METRICS
// ============================================================================

/// Record total prompt tokens processed
pub fn record_prompt_tokens_total(count: u64) {
    counter!("tokens_prompt_total").increment(count);
}

/// Record total completion tokens generated
pub fn record_completion_tokens_total(count: u64) {
    counter!("tokens_completion_total").increment(count);
}

/// Record prompt tokens per second
pub fn record_prompt_tps(tokens_per_second: f64) {
    histogram!("tokens_per_second_prompt").record(tokens_per_second);
}

/// Record decode tokens per second
pub fn record_decode_tps(tokens_per_second: f64) {
    histogram!("tokens_per_second_decode").record(tokens_per_second);
}

/// Record total tokens per request
pub fn record_tokens_per_request(sequence_id: u64, total: u64) {
    histogram!("tokens_per_request_count", "sequence_id" => sequence_id.to_string()).record(total as f64);
}

/// Record prompt throughput (tokens per second) - GAP FILL
pub fn record_prompt_throughput(sequence_id: u64, tokens_per_second: f64) {
    histogram!("prompt_throughput_tokens_per_second", "sequence_id" => sequence_id.to_string()).record(tokens_per_second);
}

// Decode throughput is already defined above with model parameter
// This is a simplified version for cases where model is not available
// ============================================================================
// INFRASTRUCTURE UTILIZATION
// ============================================================================

/// Record runtime statistics
pub fn record_runtime_stats(active_tasks: u64, blocked_threads: u64, num_workers: u64) {
    gauge!("runtime_active_tasks_count").set(active_tasks as f64);
    gauge!("runtime_blocked_threads_count").set(blocked_threads as f64);
    gauge!("runtime_num_workers_count").set(num_workers as f64);
}

/// Record worker pool utilization
pub fn record_worker_utilization(worker_id: u64, utilization: f64) {
    gauge!("worker_utilization_percent", "worker_id" => worker_id.to_string()).set(utilization);
}

// ============================================================================
// CACHE METRICS
// ============================================================================

/// Record tokenizer cache hit
pub fn record_tokenizer_cache_hit(sequence_id: u64) {
    counter!("tokenizer_cache_hits_total", "sequence_id" => sequence_id.to_string()).increment(1);
}

/// Record tokenizer cache miss
pub fn record_tokenizer_cache_miss(sequence_id: u64) {
    counter!("tokenizer_cache_misses_total", "sequence_id" => sequence_id.to_string()).increment(1);
}

/// Record tokenizer cache duration
pub fn record_tokenizer_cache_duration(sequence_id: u64, duration_seconds: f64) {
    histogram!("tokenizer_cache_duration_seconds", "sequence_id" => sequence_id.to_string()).record(duration_seconds);
}

// ============================================================================
// INITIALIZATION
// ============================================================================

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
    metrics::describe_histogram!(
        "http_request_size_bytes",
        "Size of HTTP requests in bytes"
    );
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

    // Engine phase metrics
    metrics::describe_histogram!(
        "engine_prepare_step_duration_seconds",
        "Duration of prepare_step phase"
    );
    metrics::describe_histogram!(
        "engine_run_forward_duration_seconds",
        "Duration of run_forward phase (model execution)"
    );
    metrics::describe_histogram!(
        "engine_finish_step_duration_seconds",
        "Duration of finish_step phase"
    );

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

    // TTFT (Time To First Token) metrics
    metrics::describe_histogram!("ttft_seconds", "Time to first token in seconds per sequence");
    metrics::describe_histogram!("ttft_percentile_seconds", "TTFT percentiles (p50, p99, p999)");

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

    // GAP FILL: New metrics descriptions
    metrics::describe_histogram!("prompt_throughput_tokens_per_second", "Prompt throughput in tokens per second");
    metrics::describe_histogram!("decode_throughput_tokens_per_second", "Decode throughput in tokens per second");
    metrics::describe_gauge!("metrics_collection_interval_ms", "Metrics collection interval in milliseconds");
    metrics::describe_histogram!("pd_prefill_throughput_tokens_per_second", "PD prefill throughput in tokens per second");
    metrics::describe_histogram!("kvcache_transfer_duration_seconds", "KV cache transfer duration in seconds");
    metrics::describe_gauge!("mamba_slot_usage_percent", "Mamba slot usage percentage");
    metrics::describe_gauge!("mamba_memory_used_bytes", "Mamba memory used in bytes");
    metrics::describe_counter!("gpu_metrics_errors_total", "GPU metrics collection errors by source and error type");

    // Request-level KPI tracking metrics
    metrics::describe_histogram!("request_ttft_seconds", "Time to first token in seconds per request");
    metrics::describe_histogram!("request_ttft_percentile_seconds", "TTFT percentiles (p50, p95, p99) per request");
    metrics::describe_histogram!("request_prefill_tokens_per_second", "Prefill throughput in tokens per second per request");
    metrics::describe_histogram!("request_decode_tokens_per_second", "Decode throughput in tokens per second per request");
    metrics::describe_gauge!("active_session_count", "Current number of active sessions");
    metrics::describe_gauge!("session_queue_length", "Number of sessions waiting in queue");
    metrics::describe_gauge!("session_resource_sharing_efficiency", "Resource sharing efficiency per session");
    metrics::describe_gauge!("block_cache_utilization_percent", "Block cache utilization percent");
    metrics::describe_histogram!("prefix_cache_hit_pattern", "Prefix cache hit ratio pattern per sequence");
    metrics::describe_histogram!("eviction_efficiency", "Cache eviction efficiency ratio");
    metrics::describe_gauge!("gpu_memory_bandwidth_percent", "GPU memory bandwidth utilization percent");
    metrics::describe_histogram!("gpu_power_efficiency_tokens_per_watt", "GPU power efficiency in tokens per watt");
    metrics::describe_histogram!("cpu_swap_bytes_per_second", "CPU swap rate in bytes per second");
    metrics::describe_histogram!("kv_cache_fragmentation_ratio", "KV cache fragmentation ratio");
}
// ============================================================================
// TOKENIZER METRICS
// ============================================================================

/// Record tokenizer encode duration
pub fn record_tokenizer_encode_duration(sequence_id: u64, duration_seconds: f64) {
    histogram!("tokenizer_encode_duration_seconds", "sequence_id" => sequence_id.to_string()).record(duration_seconds);
}

/// Record tokenizer decode duration
pub fn record_tokenizer_decode_duration(sequence_id: u64, duration_seconds: f64) {
    histogram!("tokenizer_decode_duration_seconds", "sequence_id" => sequence_id.to_string()).record(duration_seconds);
}

/// Record tokenizer encode token count
pub fn record_tokenizer_encode_tokens(sequence_id: u64, token_count: u64) {
    histogram!("tokenizer_encode_tokens_count", "sequence_id" => sequence_id.to_string()).record(token_count as f64);
}

/// Record tokenizer decode token count
pub fn record_tokenizer_decode_tokens(sequence_id: u64, token_count: u64) {
    histogram!("tokenizer_decode_tokens_count", "sequence_id" => sequence_id.to_string()).record(token_count as f64);
}

// ============================================================================
// GRAMMAR METRICS
// ============================================================================

/// Record grammar build duration
pub fn record_grammar_build_duration(sequence_id: u64, duration_seconds: f64) {
    histogram!("grammar_build_duration_seconds", "sequence_id" => sequence_id.to_string()).record(duration_seconds);
}

/// Record grammar compose duration
pub fn record_grammar_compose_duration(sequence_id: u64, duration_seconds: f64) {
    histogram!("grammar_compose_duration_seconds", "sequence_id" => sequence_id.to_string()).record(duration_seconds);
}

/// Record grammar constraint count
pub fn record_grammar_constraints_count(sequence_id: u64, count: u64) {
    gauge!("grammar_constraints_count", "sequence_id" => sequence_id.to_string()).set(count as f64);
}

// ============================================================================
// GUIDANCE METRICS
// ============================================================================

/// Record guidance state creation duration
pub fn record_guidance_state_creation_duration(sequence_id: u64, duration_seconds: f64) {
    histogram!("guidance_state_creation_duration_seconds", "sequence_id" => sequence_id.to_string()).record(duration_seconds);
}

/// Record guidance mask computation duration
pub fn record_guidance_mask_computation_duration(sequence_id: u64, duration_seconds: f64) {
    histogram!("guidance_mask_computation_duration_seconds", "sequence_id" => sequence_id.to_string()).record(duration_seconds);
}

/// Record guidance token validation duration
pub fn record_guidance_token_validation_duration(sequence_id: u64, duration_seconds: f64) {
    histogram!("guidance_token_validation_duration_seconds", "sequence_id" => sequence_id.to_string()).record(duration_seconds);
}

/// Record guidance committed tokens count
pub fn record_guidance_committed_tokens_count(sequence_id: u64, count: u64) {
    histogram!("guidance_committed_tokens_count", "sequence_id" => sequence_id.to_string()).record(count as f64);
}

// ============================================================================
// SAMPLING METRICS
// ============================================================================

/// Record sampling duration
pub fn record_sampling_duration(sequence_id: u64, duration_seconds: f64) {
    histogram!("sampling_duration_seconds", "sequence_id" => sequence_id.to_string()).record(duration_seconds);
}

/// Record sampling strategy
pub fn record_sampling_strategy(sequence_id: u64, strategy: &str) {
    counter!("sampling_strategy_total", "sequence_id" => sequence_id.to_string(), "strategy" => strategy.to_string()).increment(1);
}

/// Record sampling tokens count
pub fn record_sampling_tokens_count(sequence_id: u64, count: u64) {
    histogram!("sampling_tokens_count", "sequence_id" => sequence_id.to_string()).record(count as f64);
}

/// Record sampling apply repeat penalty duration
pub fn record_sampling_apply_repeat_penalty_duration(sequence_id: u64, duration_seconds: f64) {
    histogram!("sampling_apply_repeat_penalty_duration_seconds", "sequence_id" => sequence_id.to_string()).record(duration_seconds);
}

/// Record metrics collection interval - GAP FILL
pub fn record_metrics_collection_interval(interval_ms: u64) {
    gauge!("metrics_collection_interval_ms").set(interval_ms as f64);
}

// ============================================================================
// SEQUENCE PREFILL/DECODE METRICS (with cached vs new token breakdown)
// ============================================================================

/// Record per-sequence prefill with cached token breakdown
pub fn record_sequence_prefill_with_cache(
    model: &str,
    sequence_id: u64,
    total_tokens: u64,
    cached_tokens: u64,
    new_tokens: u64,
    duration_seconds: f64,
) {
    histogram!("sequence_prefill_duration_seconds", "model" => model.to_string(), "sequence_id" => sequence_id.to_string()).record(duration_seconds);
    histogram!("sequence_prefill_total_tokens", "model" => model.to_string(), "sequence_id" => sequence_id.to_string()).record(total_tokens as f64);
    histogram!("sequence_prefill_cached_tokens", "model" => model.to_string(), "sequence_id" => sequence_id.to_string()).record(cached_tokens as f64);
    histogram!("sequence_prefill_new_tokens", "model" => model.to_string(), "sequence_id" => sequence_id.to_string()).record(new_tokens as f64);
    gauge!("sequence_prefill_cached_ratio", "model" => model.to_string(), "sequence_id" => sequence_id.to_string()).set(if total_tokens > 0 { cached_tokens as f64 / total_tokens as f64 } else { 0.0 });
}

/// Record per-sequence decode with cached token breakdown
pub fn record_sequence_decode_with_cache(
    model: &str,
    sequence_id: u64,
    total_tokens: u64,
    cached_tokens: u64,
    new_tokens: u64,
    duration_seconds: f64,
) {
    histogram!("sequence_decode_duration_seconds", "model" => model.to_string(), "sequence_id" => sequence_id.to_string()).record(duration_seconds);
    histogram!("sequence_decode_total_tokens", "model" => model.to_string(), "sequence_id" => sequence_id.to_string()).record(total_tokens as f64);
    histogram!("sequence_decode_cached_tokens", "model" => model.to_string(), "sequence_id" => sequence_id.to_string()).record(cached_tokens as f64);
    histogram!("sequence_decode_new_tokens", "model" => model.to_string(), "sequence_id" => sequence_id.to_string()).record(new_tokens as f64);
    gauge!("sequence_decode_cached_ratio", "model" => model.to_string(), "sequence_id" => sequence_id.to_string()).set(if total_tokens > 0 { cached_tokens as f64 / total_tokens as f64 } else { 0.0 });
}

/// Record per-sequence total tokens (prompt + completion)
pub fn record_sequence_total_tokens(
    model: &str,
    sequence_id: u64,
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
) {
    histogram!("sequence_prompt_tokens", "model" => model.to_string(), "sequence_id" => sequence_id.to_string()).record(prompt_tokens as f64);
    histogram!("sequence_completion_tokens", "model" => model.to_string(), "sequence_id" => sequence_id.to_string()).record(completion_tokens as f64);
    histogram!("sequence_total_tokens", "model" => model.to_string(), "sequence_id" => sequence_id.to_string()).record(total_tokens as f64);
}

/// Record per-sequence token generation rate
pub fn record_sequence_tokens_per_second(
    model: &str,
    sequence_id: u64,
    tokens_per_second: f64,
) {
    histogram!("sequence_tokens_per_second", "model" => model.to_string(), "sequence_id" => sequence_id.to_string()).record(tokens_per_second);
}

// ============================================================================
// REQUEST-LEVEL METRICS (no sequence_id available)
// ============================================================================

/// Record sequence added (request-level metric)
pub fn record_sequence_added_request() {
    counter!("sequence_added_total").increment(1);
}

/// Record sequence completed (request-level metric)
pub fn record_sequence_completed_request(status: &str) {
    counter!("sequence_completed_total", "status" => status.to_string()).increment(1);
}

/// Record reasoning enabled (request-level metric)
/// Also records to reasoning_blocks_enabled_total for backward compatibility
pub fn record_reasoning_enabled_request() {
    counter!("reasoning_enabled_total").increment(1);
    counter!("reasoning_blocks_enabled_total").increment(1);
}

/// Record mamba slot usage percent (aggregate metric)
pub fn record_mamba_slot_usage_percent_aggregate(percent: f64) {
    gauge!("mamba_slot_usage_percent").set(percent);
}

/// Record mamba memory used bytes (aggregate metric)
pub fn record_mamba_memory_used_bytes_aggregate(bytes: u64) {
    gauge!("mamba_memory_used_bytes").set(bytes as f64);
}

// ============================================================================//
// REQUEST-LEVEL KPI TRACKING (for adaptive tuning)
// ============================================================================//

/// Record request-level TTFT (Time To First Token)
pub fn record_request_ttft(sequence_id: u64, ttft_seconds: f64) {
    histogram!("request_ttft_seconds", "sequence_id" => sequence_id.to_string()).record(ttft_seconds);
}

/// Record request-level TTFT percentile (p50, p95, p99)
pub fn record_request_ttft_percentile(percentile: &str, duration_seconds: f64) {
    histogram!("request_ttft_percentile_seconds", "percentile" => percentile.to_string()).record(duration_seconds);
}

/// Record request-level prefill throughput (tokens/second)
pub fn record_request_prefill_tps(sequence_id: u64, tokens_per_second: f64) {
    histogram!("request_prefill_tokens_per_second", "sequence_id" => sequence_id.to_string()).record(tokens_per_second);
}

/// Record request-level decode throughput (tokens/second)
pub fn record_request_decode_tps(sequence_id: u64, tokens_per_second: f64) {
    histogram!("request_decode_tokens_per_second", "sequence_id" => sequence_id.to_string()).record(tokens_per_second);
}

/// Record active session count
pub fn record_active_sessions(count: u64) {
    gauge!("active_session_count").set(count as f64);
}

/// Record session queue length
pub fn record_session_queue_length(count: u64) {
    gauge!("session_queue_length").set(count as f64);
}

/// Record session resource sharing efficiency
pub fn record_session_resource_sharing(session_id: &str, efficiency: f64) {
    gauge!("session_resource_sharing_efficiency", "session_id" => session_id.to_string()).set(efficiency);
}

/// Record block-level cache utilization
pub fn record_block_cache_utilization(device: &str, utilization: f64) {
    gauge!("block_cache_utilization_percent", "device" => device.to_string()).set(utilization);
}

/// Record prefix cache hit pattern for a sequence
pub fn record_prefix_cache_hit_pattern(sequence_id: u64, hit_ratio: f64) {
    histogram!("prefix_cache_hit_pattern", "sequence_id" => sequence_id.to_string()).record(hit_ratio);
}

/// Record eviction efficiency
pub fn record_eviction_efficiency(device: &str, efficiency: f64) {
    histogram!("eviction_efficiency", "device" => device.to_string()).record(efficiency);
}

/// Record GPU memory bandwidth utilization
#[cfg(feature = "cuda")]
pub fn record_gpu_memory_bandwidth_percent(device_id: &str, percent: f64) {
    gauge!("gpu_memory_bandwidth_percent", "device_id" => device_id.to_string()).set(percent);
}

/// Record GPU power efficiency (tokens per watt)
#[cfg(feature = "cuda")]
pub fn record_gpu_power_efficiency(device_id: &str, tokens_per_watt: f64) {
    histogram!("gpu_power_efficiency_tokens_per_watt", "device_id" => device_id.to_string()).record(tokens_per_watt);
}

/// Record CPU memory swap rate
pub fn record_cpu_swap_rate(bytes_per_second: f64) {
    histogram!("cpu_swap_bytes_per_second").record(bytes_per_second);
}

/// Record KV cache fragmentation ratio
pub fn record_kv_cache_fragmentation(device: &str, ratio: f64) {
    histogram!("kv_cache_fragmentation_ratio", "device" => device.to_string()).record(ratio);
}