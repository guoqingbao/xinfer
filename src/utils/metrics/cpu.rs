// src/utils/metrics/cpu.rs
//! CPU and system resource monitoring module
//!
//! Provides continuous monitoring of CPU utilization, memory usage,
//! load averages, and other system-level metrics using sysinfo.
//!
//! This module is ALWAYS enabled - no feature flags required.

use metrics::gauge;
use std::time::Duration;
use sysinfo::{System, Pid, Disks, Networks};

// Metrics interval configuration
pub const METRICS_INTERVAL_ENV: &str = "VLLM_RS_METRICS_INTERVAL_MS";
pub const DEFAULT_METRICS_INTERVAL_MS: u64 = 100;

pub fn get_metrics_interval() -> Duration {
    let default = Duration::from_millis(DEFAULT_METRICS_INTERVAL_MS);
    std::env::var(METRICS_INTERVAL_ENV)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map(|ms| Duration::from_millis(ms))
        .unwrap_or(default)
}

/// Record CPU and system metrics from sysinfo
pub fn record_cpu_metrics() {
    let mut system = System::new();

    // Refresh only what we need for performance
    system.refresh_cpu_all();
    system.refresh_memory();

    // System memory metrics
    gauge!("system_memory_total_bytes").set(system.total_memory() as f64);
    gauge!("system_memory_used_bytes").set(system.used_memory() as f64);
    gauge!("system_memory_free_bytes").set(system.free_memory() as f64);
    gauge!("system_memory_available_bytes").set(system.available_memory() as f64);

    // Swap metrics
    gauge!("system_swap_total_bytes").set(system.total_swap() as f64);
    gauge!("system_swap_used_bytes").set(system.used_swap() as f64);
    gauge!("system_swap_free_bytes").set(system.free_swap() as f64);

    // Per-CPU core utilization
    for (i, cpu) in system.cpus().iter().enumerate() {
        gauge!("cpu_utilization_percent", "cpu_id" => i.to_string()).set(cpu.cpu_usage() as f64);
        gauge!("cpu_frequency_mhz", "cpu_id" => i.to_string()).set(cpu.frequency() as f64);
    }

    // Load averages (1, 5, 15 minutes)
    let load_avg = sysinfo::System::load_average();
    gauge!("system_load_avg1").set(load_avg.one);
    gauge!("system_load_avg5").set(load_avg.five);
    gauge!("system_load_avg15").set(load_avg.fifteen);

    // Thread count
    gauge!("process_thread_count").set(system.processes().len() as f64);

    // System info metrics (as labels)
    gauge!("system_uptime_seconds").set(sysinfo::System::uptime() as f64);
    gauge!("system_boot_time_seconds").set(sysinfo::System::boot_time() as f64);
    gauge!("system_physical_core_count").set(sysinfo::System::physical_core_count().unwrap_or(0) as f64);
    gauge!("system_open_files_limit").set(sysinfo::System::open_files_limit().unwrap_or(0) as f64);
}

/// Record process memory metrics for vllm-rs main process and all IPC children
pub fn record_process_memory_metrics() {
    let system = System::new_all();
    
    // Get all processes and clone the data we need
    let processes: Vec<(Pid, String, u64, u64, f32, u64, u64)> = system
        .processes()
        .iter()
        .filter_map(|(pid, process)| {
            let name = process.name().to_string_lossy().into_owned();
            // Only track vllm-rs processes and their children
            if name.contains("vllm") || name.contains("runner") {
                Some((
                    *pid,
                    name,
                    process.memory(),
                    process.virtual_memory(),
                    process.cpu_usage(),
                    process.start_time(),
                    process.run_time(),
                ))
            } else {
                None
            }
        })
        .collect();
    
    // Now record metrics with cloned data
    for (pid, name, memory, vms, cpu_usage, start_time, run_time) in processes {
        let pid_str = pid.to_string();
        gauge!("process_memory_rss_bytes", "pid" => pid_str.clone(), "name" => name.clone()).set(memory as f64);
        gauge!("process_memory_vms_bytes", "pid" => pid_str.clone(), "name" => name.clone()).set(vms as f64);
        gauge!("process_cpu_usage_percent", "pid" => pid_str.clone(), "name" => name.clone()).set(cpu_usage as f64);
        gauge!("process_start_time_seconds", "pid" => pid_str.clone(), "name" => name.clone()).set(start_time as f64);
        gauge!("process_uptime_seconds", "pid" => pid_str.clone(), "name" => name.clone()).set(run_time as f64);
    }
}

/// Record disk I/O metrics
pub fn record_disk_io_metrics() {
    let disks = Disks::new_with_refreshed_list();

    for disk in disks.iter() {
        let disk_name: String = disk.name().to_string_lossy().into_owned();
        let usage = disk.usage();
        // DiskUsage is Copy, so we can just read the values directly
        gauge!("disk_io_read_bytes_total", "disk" => disk_name.clone()).set(usage.read_bytes as f64);
        gauge!("disk_io_write_bytes_total", "disk" => disk_name.clone()).set(usage.written_bytes as f64);
        // Additional disk metrics
        gauge!("disk_total_space_bytes", "disk" => disk_name.clone()).set(disk.total_space() as f64);
        gauge!("disk_available_space_bytes", "disk" => disk_name.clone()).set(disk.available_space() as f64);
    }
}

/// Record network I/O metrics
pub fn record_network_io_metrics() {
    let networks = Networks::new_with_refreshed_list();

    for (iface_name, iface) in networks.iter() {
        gauge!("network_io_bytes_sent_total", "interface" => iface_name.clone()).set(iface.transmitted() as f64);
        gauge!("network_io_bytes_received_total", "interface" => iface_name.clone()).set(iface.received() as f64);
    }
}

/// Async task that continuously monitors CPU metrics
pub async fn run_cpu_monitor(interval: Duration) {
    loop {
        record_cpu_metrics();
        record_process_memory_metrics();
        record_disk_io_metrics();
        record_network_io_metrics();
        tokio::time::sleep(interval).await;
    }
}

/// Start CPU monitoring in background
pub fn start_cpu_monitor(interval: Duration) -> tokio::task::JoinHandle<()> {
    tokio::spawn(run_cpu_monitor(interval))
}

/// Collect all system metrics
pub fn collect_system_metrics() {
    record_cpu_metrics();
    record_process_memory_metrics();
    record_disk_io_metrics();
    record_network_io_metrics();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_metrics_interval_default() {
        // Should return default 100ms when env var is not set
        std::env::remove_var(METRICS_INTERVAL_ENV);
        let interval = get_metrics_interval();
        assert_eq!(interval.as_millis(), DEFAULT_METRICS_INTERVAL_MS as u128);
    }

    #[test]
    fn test_get_metrics_interval_custom() {
        // Should return custom value when env var is set
        std::env::set_var(METRICS_INTERVAL_ENV, "500");
        let interval = get_metrics_interval();
        assert_eq!(interval.as_millis(), 500 as u128);
        std::env::remove_var(METRICS_INTERVAL_ENV);
    }

    #[test]
    fn test_record_cpu_metrics() {
        // Should not panic
        record_cpu_metrics();
    }

    #[test]
    fn test_record_process_memory_metrics() {
        // Should not panic
        record_process_memory_metrics();
    }

    #[test]
    fn test_record_disk_io_metrics() {
        // Should not panic
        record_disk_io_metrics();
    }

    #[test]
    fn test_record_network_io_metrics() {
        // Should not panic
        record_network_io_metrics();
    }
}