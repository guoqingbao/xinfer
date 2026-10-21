#[cfg(feature = "python")]
use pyo3::prelude::*;
// The Rust programmatic API (EngineBuilder/Engine/EngineStream). Python builds
// expose the engine through the pyo3 bindings (py/mod.rs) instead, so this module
// is Rust-only. The engine logic itself (core/) compiles under every feature set.
#[cfg(not(feature = "python"))]
pub mod api;
pub mod core;
pub mod domain;
pub mod mcp;
pub mod models;
#[cfg(feature = "python")]
pub mod py;
pub mod runner;
pub mod server;
pub mod speculative;
pub mod tools;
pub mod transfer;
pub mod utils;
#[cfg(feature = "python")]
use crate::core::GenerationOutput;
#[cfg(feature = "python")]
use crate::py::Engine;
#[cfg(feature = "python")]
use crate::transfer::{PdConfig, PdMethod, PdRole};
#[cfg(feature = "python")]
use crate::utils::chat_template::Message;
#[cfg(feature = "python")]
use crate::utils::config::{EngineConfig, GenerationConfig, SamplingParams};
/// A Python module implemented in Rust.
#[cfg(feature = "python")]
#[pymodule]
fn xinfer(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Engine>()?;
    m.add_class::<EngineConfig>()?;
    m.add_class::<SamplingParams>()?;
    m.add_class::<GenerationConfig>()?;
    m.add_class::<GenerationOutput>()?;
    m.add_class::<Message>()?;
    m.add_class::<PdConfig>()?;
    m.add_class::<PdMethod>()?;
    m.add_class::<PdRole>()?;
    Ok(())
}
