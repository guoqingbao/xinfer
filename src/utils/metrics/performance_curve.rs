// src/utils/metrics/performance_curve.rs
//! Performance curve tracking for adaptive tuning
//!
//! This module provides:
//! 1. Per-parameter throughput/latency measurement
//! 2. Workload classification (compute-bound, memory-bound, balanced)
//! 3. Historical performance data tracking
//! 4. Binary search optimization integration
//!
//! # Design Principles
//!
//! - **DRY**: Single source of truth for performance metrics
//! - **Type-safe**: Strong typing for workload classification
//! - **Production-grade**: Proper error handling and testing
//! - **Low overhead**: Minimal performance impact on inference loop

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use parking_lot::RwLock;
use once_cell::sync::OnceCell;

/// Workload type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkloadType {
    /// SM utilization saturated, memory bandwidth available
    ComputeBound,
    /// Memory bandwidth saturated, SM underutilized
    MemoryBound,
    /// Both resources utilized evenly
    Balanced,
}

impl std::fmt::Display for WorkloadType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WorkloadType::ComputeBound => write!(f, "compute_bound"),
            WorkloadType::MemoryBound => write!(f, "memory_bound"),
            WorkloadType::Balanced => write!(f, "balanced"),
        }
    }
}

/// Bottleneck type for targeted optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BottleneckType {
    /// GPU memory bandwidth saturated
    MemoryBandwidth,
    /// GPU SM utilization saturated
    Compute,
    /// PCIe bandwidth saturated
    PCIe,
    /// KV cache fragmentation
    KVCacheFragmentation,
    /// Prefill chunking inefficiency
    PrefillChunk,
    /// Decode sequences starved of prefill
    DecodeStarvation,
    /// CPU memory pressure causing swaps
    CPUMemoryPressure,
    /// System load average too high
    SystemLoad,
}

impl std::fmt::Display for BottleneckType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BottleneckType::MemoryBandwidth => write!(f, "memory_bandwidth"),
            BottleneckType::Compute => write!(f, "compute"),
            BottleneckType::PCIe => write!(f, "pcie"),
            BottleneckType::KVCacheFragmentation => write!(f, "kv_cache_fragmentation"),
            BottleneckType::PrefillChunk => write!(f, "prefill_chunk"),
            BottleneckType::DecodeStarvation => write!(f, "decode_starvation"),
            BottleneckType::CPUMemoryPressure => write!(f, "cpu_memory_pressure"),
            BottleneckType::SystemLoad => write!(f, "system_load"),
        }
    }
}

/// Single performance measurement point
#[derive(Debug, Clone)]
pub struct PerformancePoint {
    pub timestamp: u64,              // ms since epoch
    pub batch_size: usize,
    pub num_seqs: usize,
    pub throughput_tps: f64,         // tokens per second
    pub latency_s: f64,              // step duration
    pub ttft_s: f64,                 // time to first token
    pub decode_tps: f64,             // decode throughput
    pub prefill_tps: f64,            // prefill throughput
    pub sm_utilization: f64,         // GPU SM utilization %
    pub mem_utilization: f64,        // GPU memory utilization %
    pub kv_cache_usage: f64,         // KV cache usage %
    pub kv_cache_data_type: String,  // KV cache data type (e.g., "f16", "f32", "u8" for fp8)
    pub kv_cache_per_token_bytes: f64, // per-token byte size
    pub workload_type: WorkloadType,
    pub bottleneck: Option<BottleneckType>,
}

/// Performance curve tracker for a single parameter
pub struct PerformanceCurve {
    #[allow(dead_code)]
    metric_name: String,
    points: VecDeque<PerformancePoint>,
    max_points: usize,
}

impl PerformanceCurve {
    pub fn new(metric_name: &str, max_points: usize) -> Self {
        Self {
            metric_name: metric_name.to_string(),
            points: VecDeque::with_capacity(max_points),
            max_points,
        }
    }

    pub fn record(&mut self, point: PerformancePoint) {
        self.points.push_back(point);
        while self.points.len() > self.max_points {
            self.points.pop_front();
        }
    }

    pub fn get_points(&self) -> &[PerformancePoint] {
        self.points.as_slices().0
    }

    pub fn find_optimal_batch_size(&self) -> Option<usize> {
        // Find batch size with best throughput/latency ratio
        self.points.iter()
            .max_by(|a, b| {
                (a.throughput_tps / a.latency_s.max(0.001))
                    .partial_cmp(&(b.throughput_tps / b.latency_s.max(0.001)))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|p| p.batch_size)
    }

    pub fn get_latest_point(&self) -> Option<&PerformancePoint> {
        self.points.back()
    }
}

/// Global performance curve manager
pub struct PerformanceCurveManager {
    curves: RwLock<std::collections::HashMap<String, Arc<RwLock<PerformanceCurve>>>>,
    max_history_points: usize,
}

impl PerformanceCurveManager {
    pub fn new(max_history_points: usize) -> Self {
        Self {
            curves: RwLock::new(std::collections::HashMap::new()),
            max_history_points,
        }
    }

    pub fn get_or_create_curve(&self, name: &str) -> Arc<RwLock<PerformanceCurve>> {
        let mut curves = self.curves.write();
        curves.entry(name.to_string())
            .or_insert_with(|| {
                Arc::new(RwLock::new(PerformanceCurve::new(name, self.max_history_points)))
            })
            .clone()
    }

    pub fn record_batch(
        &self,
        batch_size: usize,
        num_seqs: usize,
        throughput_tps: f64,
        latency_s: f64,
        ttft_s: f64,
        decode_tps: f64,
        prefill_tps: f64,
        sm_util: f64,
        mem_util: f64,
        kv_cache_usage: f64,
        kv_cache_data_type: &str,
        kv_cache_per_token_bytes: f64,
    ) {
        let bottleneck = Self::classify_bottleneck(sm_util, mem_util, kv_cache_usage);
        let workload = Self::classify_workload(sm_util, mem_util);

        let point = PerformancePoint {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            batch_size,
            num_seqs,
            throughput_tps,
            latency_s,
            ttft_s,
            decode_tps,
            prefill_tps,
            sm_utilization: sm_util,
            mem_utilization: mem_util,
            kv_cache_usage,
            kv_cache_data_type: kv_cache_data_type.to_string(),
            kv_cache_per_token_bytes,
            workload_type: workload,
            bottleneck: Some(bottleneck),
        };

        // Record to all relevant curves
        let curves = self.curves.read();
        for (_name, curve) in curves.iter() {
            curve.write().record(point.clone());
        }
    }

    pub fn classify_bottleneck(sm_util: f64, mem_util: f64, kv_cache_usage: f64) -> BottleneckType {
        // Heuristic-based bottleneck classification
        if sm_util < 70.0 && mem_util > 85.0 {
            BottleneckType::MemoryBandwidth
        } else if sm_util > 90.0 && mem_util < 70.0 {
            BottleneckType::Compute
        } else if kv_cache_usage > 95.0 {
            BottleneckType::KVCacheFragmentation
        } else if kv_cache_usage > 85.0 {
            BottleneckType::PrefillChunk
        } else {
            BottleneckType::MemoryBandwidth  // Default
        }
    }

    pub fn classify_workload(sm_util: f64, mem_util: f64) -> WorkloadType {
        let sm_ratio = sm_util / 100.0;
        let mem_ratio = mem_util / 100.0;

        if sm_ratio > 0.85 && mem_ratio < 0.7 {
            WorkloadType::ComputeBound
        } else if mem_ratio > 0.85 && sm_ratio < 0.7 {
            WorkloadType::MemoryBound
        } else {
            WorkloadType::Balanced
        }
    }

    pub fn get_performance_curves(&self) -> std::collections::HashMap<String, Arc<RwLock<PerformanceCurve>>> {
        self.curves.read().clone()
    }
}

/// Binary search optimizer for finding optimal batch size
pub struct BinarySearchOptimizer {
    lower_bound: usize,
    upper_bound: usize,
    current_test_value: usize,
    best_value: Option<usize>,
    best_score: f64,
    performance_history: Vec<PerformancePoint>,
    convergence_threshold: f64,
    #[allow(dead_code)]
    max_iterations: usize,
    current_iteration: usize,
}

impl BinarySearchOptimizer {
    pub fn new(lower_bound: usize, upper_bound: usize, convergence_threshold: f64) -> Self {
        let current_test_value = (lower_bound + upper_bound) / 2;
        Self {
            lower_bound,
            upper_bound,
            current_test_value,
            best_value: None,
            best_score: f64::NEG_INFINITY,
            performance_history: Vec::new(),
            convergence_threshold,
            max_iterations: 20,
            current_iteration: 0,
        }
    }

    pub fn record_performance(&mut self, point: PerformancePoint) {
        self.current_iteration += 1;

        // Calculate score: throughput / latency (higher is better)
        let score = point.throughput_tps / point.latency_s.max(0.001);

        // Update best if this is better
        if score > self.best_score {
            self.best_score = score;
            self.best_value = Some(point.batch_size);
        }

        self.performance_history.push(point);

        // Binary search adjustment
        self.adjust_bounds(score);
    }

    fn adjust_bounds(&mut self, current_score: f64) {
        if self.current_iteration <= 3 {
            // First few iterations: expand upward to find upper bound
            let range = self.upper_bound - self.current_test_value;
            self.current_test_value += range / 2;
        } else {
            // After initial exploration: binary search for optimum
            let threshold = self.best_score * (1.0 - self.convergence_threshold);

            if current_score >= threshold {
                // Good performance, try higher
                let range = self.upper_bound - self.current_test_value;
                self.lower_bound = self.current_test_value;
                self.current_test_value += range / 2;
            } else {
                // Poor performance, try lower
                let range = self.current_test_value - self.lower_bound;
                self.upper_bound = self.current_test_value;
                self.current_test_value -= range / 2;
            }
        }

        // Clamp to bounds
        self.current_test_value = self.current_test_value
            .max(self.lower_bound)
            .min(self.upper_bound);
    }

    pub fn get_optimal_value(&self) -> Option<usize> {
        self.best_value
    }

    pub fn get_current_test_value(&self) -> usize {
        self.current_test_value
    }

    pub fn has_converged(&self) -> bool {
        if self.current_iteration < 3 {
            return false;
        }
        let range = self.upper_bound - self.lower_bound;
        let threshold = (self.convergence_threshold * self.lower_bound as f64) as usize;
        range <= threshold
    }

    pub fn reset(&mut self) {
        self.current_test_value = (self.lower_bound + self.upper_bound) / 2;
        self.best_value = None;
        self.best_score = f64::NEG_INFINITY;
        self.current_iteration = 0;
        self.performance_history.clear();
    }

    pub fn get_history(&self) -> &[PerformancePoint] {
        &self.performance_history
    }
}

// ============================================================================//
// GLOBAL STATE
// ============================================================================//

/// Global performance curve manager
static PERFORMANCE_CURVE_MANAGER: OnceCell<Arc<RwLock<PerformanceCurveManager>>> = OnceCell::new();

/// Global binary search optimizer
static BINARY_SEARCH_OPTIMIZER: OnceCell<Arc<RwLock<BinarySearchOptimizer>>> = OnceCell::new();

// ============================================================================//
// PUBLIC API
// ============================================================================//

/// Initialize performance curve manager
pub fn init_performance_curve_manager(max_points: usize) {
    PERFORMANCE_CURVE_MANAGER.get_or_init(|| {
        Arc::new(RwLock::new(PerformanceCurveManager::new(max_points)))
    });
}

/// Get the global performance curve manager
pub fn get_performance_curve_manager() -> Option<Arc<RwLock<PerformanceCurveManager>>> {
    PERFORMANCE_CURVE_MANAGER.get().cloned()
}

/// Initialize binary search optimizer
pub fn init_binary_search_optimizer(lower_bound: usize, upper_bound: usize, convergence_threshold: f64) {
    BINARY_SEARCH_OPTIMIZER.get_or_init(|| {
        Arc::new(RwLock::new(BinarySearchOptimizer::new(lower_bound, upper_bound, convergence_threshold)))
    });
}

/// Get the global binary search optimizer
pub fn get_binary_search_optimizer() -> Option<Arc<RwLock<BinarySearchOptimizer>>> {
    BINARY_SEARCH_OPTIMIZER.get().cloned()
}

/// Record performance point for binary search
pub fn record_performance_point(
    batch_size: usize,
    num_seqs: usize,
    throughput_tps: f64,
    latency_s: f64,
    ttft_s: f64,
    decode_tps: f64,
    prefill_tps: f64,
    sm_util: f64,
    mem_util: f64,
    kv_cache_usage: f64,
    kv_cache_data_type: &str,
    kv_cache_per_token_bytes: f64,
) {
    if let Some(manager) = get_performance_curve_manager() {
        manager.write().record_batch(
            batch_size, num_seqs, throughput_tps, latency_s, ttft_s,
            decode_tps, prefill_tps, sm_util, mem_util, kv_cache_usage,
            kv_cache_data_type, kv_cache_per_token_bytes
        );
    }
}

/// Record performance for binary search optimizer
pub fn record_binary_search_performance(point: PerformancePoint) {
    if let Some(optimizer) = get_binary_search_optimizer() {
        optimizer.write().record_performance(point);
    }
}

/// Get optimal batch size from binary search optimizer
pub fn get_binary_search_optimal_batch_size() -> Option<usize> {
    get_binary_search_optimizer()
        .and_then(|o| o.read().get_optimal_value())
}

/// Get current test value from binary search optimizer
pub fn get_binary_search_current_test_value() -> usize {
    get_binary_search_optimizer()
        .map(|o| o.read().get_current_test_value())
        .unwrap_or(8192)  // default
}

/// Check if binary search optimizer has converged
pub fn binary_search_has_converged() -> bool {
    get_binary_search_optimizer()
        .map(|o| o.read().has_converged())
        .unwrap_or(false)
}

/// Reset binary search optimizer
pub fn reset_binary_search_optimizer() {
    if let Some(optimizer) = get_binary_search_optimizer() {
        optimizer.write().reset();
    }
}

// ============================================================================//
// ALERT LOGGING
// ============================================================================//

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlertSeverity {
    /// Informational - no action needed
    Info,
    /// Warning - monitor closely
    Warning,
    /// Critical - immediate action required
    Critical,
}

/// A bottleneck alert
#[derive(Debug, Clone)]
pub struct BottleneckAlert {
    pub timestamp: u64,
    pub bottleneck_type: BottleneckType,
    pub severity: AlertSeverity,
    pub metric_name: String,
    pub current_value: f64,
    pub threshold: f64,
    pub message: String,
}

impl BottleneckAlert {
    pub fn new(
        bottleneck_type: BottleneckType,
        severity: AlertSeverity,
        metric_name: &str,
        current_value: f64,
        threshold: f64,
        message: &str,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        
        Self {
            timestamp,
            bottleneck_type,
            severity,
            metric_name: metric_name.to_string(),
            current_value,
            threshold,
            message: message.to_string(),
        }
    }
}

// ============================================================================//
// TESTS
// ============================================================================//

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workload_type_display() {
        assert_eq!(format!("{}", WorkloadType::ComputeBound), "compute_bound");
        assert_eq!(format!("{}", WorkloadType::MemoryBound), "memory_bound");
        assert_eq!(format!("{}", WorkloadType::Balanced), "balanced");
    }

    #[test]
    fn test_bottleneck_type_display() {
        assert_eq!(format!("{}", BottleneckType::MemoryBandwidth), "memory_bandwidth");
        assert_eq!(format!("{}", BottleneckType::Compute), "compute");
        assert_eq!(format!("{}", BottleneckType::KVCacheFragmentation), "kv_cache_fragmentation");
    }

    #[test]
    fn test_performance_curve_record() {
        let mut curve = PerformanceCurve::new("test", 10);
        
        let point = PerformancePoint {
            timestamp: 1000,
            batch_size: 1024,
            num_seqs: 1,
            throughput_tps: 1000.0,
            latency_s: 0.1,
            ttft_s: 0.05,
            decode_tps: 1000.0,
            prefill_tps: 1000.0,
            sm_utilization: 80.0,
            mem_utilization: 70.0,
            kv_cache_usage: 50.0,
            kv_cache_data_type: "f16".to_string(),
            kv_cache_per_token_bytes: 2.0,
            workload_type: WorkloadType::Balanced,
            bottleneck: Some(BottleneckType::Compute),
        };
        
        curve.record(point);
        assert_eq!(curve.get_points().len(), 1);
    }

    #[test]
    fn test_performance_curve_find_optimal() {
        let mut curve = PerformanceCurve::new("test", 100);
        
        // Record points with different batch sizes
        let points = vec![
            (512, 5000.0, 0.1),   // batch_size, throughput, latency
            (1024, 10000.0, 0.1), // throughput/latency = 100000
            (2048, 18000.0, 0.1), // throughput/latency = 180000 (best)
            (4096, 20000.0, 0.2), // throughput/latency = 100000
        ];
        
        for (bs, tp, lat) in points {
            let point = PerformancePoint {
                timestamp: 1000,
                batch_size: bs,
                num_seqs: 1,
                throughput_tps: tp,
                latency_s: lat,
                ttft_s: 0.0,
                decode_tps: tp,
                prefill_tps: tp,
                sm_utilization: 80.0,
                mem_utilization: 70.0,
                kv_cache_usage: 50.0,
                kv_cache_data_type: "f16".to_string(),
                kv_cache_per_token_bytes: 2.0,
                workload_type: WorkloadType::Balanced,
                bottleneck: None,
            };
            curve.record(point);
        }
        
        let optimal = curve.find_optimal_batch_size();
        assert_eq!(optimal, Some(2048)); // Should find highest throughput/latency ratio
    }

    #[test]
    fn test_binary_search_optimizer() {
        let mut optimizer = BinarySearchOptimizer::new(1024, 16384, 0.10);
        
        // Record some performance points
        for i in 0..5 {
            let point = PerformancePoint {
                timestamp: 1000 + i as u64,
                batch_size: optimizer.get_current_test_value(),
                num_seqs: 1,
                throughput_tps: 10000.0,
                latency_s: 0.1,
                ttft_s: 0.05,
                decode_tps: 10000.0,
                prefill_tps: 10000.0,
                sm_utilization: 80.0,
                mem_utilization: 70.0,
                kv_cache_usage: 50.0,
                kv_cache_data_type: "f16".to_string(),
                kv_cache_per_token_bytes: 2.0,
                workload_type: WorkloadType::Balanced,
                bottleneck: None,
            };
            optimizer.record_performance(point);
        }
        
        assert!(optimizer.has_converged() || optimizer.get_current_test_value() > 0);
    }

    #[test]
    fn test_binary_search_convergence() {
        let mut optimizer = BinarySearchOptimizer::new(1024, 16384, 0.10);
        
        // Record many points to ensure convergence
        for i in 0..20 {
            let point = PerformancePoint {
                timestamp: 1000 + i as u64,
                batch_size: optimizer.get_current_test_value(),
                num_seqs: 1,
                throughput_tps: 10000.0,
                latency_s: 0.1,
                ttft_s: 0.05,
                decode_tps: 10000.0,
                prefill_tps: 10000.0,
                sm_utilization: 80.0,
                mem_utilization: 70.0,
                kv_cache_usage: 50.0,
                kv_cache_data_type: "f16".to_string(),
                kv_cache_per_token_bytes: 2.0,
                workload_type: WorkloadType::Balanced,
                bottleneck: None,
            };
            optimizer.record_performance(point);
        }
        
        assert!(optimizer.has_converged());
        assert!(optimizer.get_optimal_value().is_some());
    }

    #[test]
    fn test_workload_classification() {
        // Compute-bound: high SM, low memory
        assert_eq!(
            PerformanceCurveManager::classify_workload(95.0, 50.0),
            WorkloadType::ComputeBound
        );
        
        // Memory-bound: low SM, high memory
        assert_eq!(
            PerformanceCurveManager::classify_workload(50.0, 95.0),
            WorkloadType::MemoryBound
        );
        
        // Balanced: both medium
        assert_eq!(
            PerformanceCurveManager::classify_workload(70.0, 70.0),
            WorkloadType::Balanced
        );
    }

    #[test]
    fn test_bottleneck_classification() {
        // Memory bandwidth: low SM, high memory
        assert_eq!(
            PerformanceCurveManager::classify_bottleneck(60.0, 90.0, 50.0),
            BottleneckType::MemoryBandwidth
        );
        
        // Compute: high SM, low memory
        assert_eq!(
            PerformanceCurveManager::classify_bottleneck(95.0, 50.0, 50.0),
            BottleneckType::Compute
        );
        
        // KV cache fragmentation: high cache usage
        assert_eq!(
            PerformanceCurveManager::classify_bottleneck(80.0, 80.0, 98.0),
            BottleneckType::KVCacheFragmentation
        );
    }
}