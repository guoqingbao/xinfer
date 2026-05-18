# Rust crate usage

This crate exposes a Rust-facing API for loading models, running generation, and optionally running
an OpenAI-compatible service without changing the existing project structure.

## Add dependency

```toml
[dependencies]
vllm-rs = { git = "https://github.com/guoqingbao/vllm.rs.git", rev = "1377fa9" }

[features]
cuda = ["vllm_rs/cuda"]
```

Use the same Cargo features you would use for the CLI (`cuda`, `metal`, `nccl`, etc.).

## Direct generation (text)

```rust
use vllm_rs::api::{EngineBuilder, ModelRepo};
use vllm_rs::server::{ChatMessage, MessageContentType};
use vllm_rs::utils::{config::SamplingParams, log_throughput};

fn main() -> anyhow::Result<()> {
    let mut engine =
        EngineBuilder::new(ModelRepo::ModelID(("google/gemma-3-4b-it", None))).build()?;

    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: MessageContentType::PureText("Say hello from the Rust API.".to_string()),
    }];

    let params = SamplingParams::default();
    let output = engine.generate(params, messages)?;
    println!("\n\n{}", output.decode_output);

    log_throughput(&vec![output]);
}
```

## Multimodal request (URL or base64)

```rust
use vllm_rs::api::{EngineBuilder, ModelRepo};
use vllm_rs::server::{ChatMessage, MessageContent, MessageContentType};
use vllm_rs::utils::config::SamplingParams;

fn main() -> candle_core::Result<()> {
    let mut engine = EngineBuilder::new(ModelRepo::ModelID((
        "Qwen/Qwen3-VL-8B-Instruct".to_string(),
        None,
    )))
    .build()?;

    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: MessageContentType::Multi(vec![
            MessageContent::Text {
                text: "Describe this image:".to_string(),
            },
            MessageContent::ImageUrl {
                image_url: "https://example.com/cat.png".to_string(),
            },
        ]),
    }];

    let params = SamplingParams::default();
    let output = engine.generate(params, messages)?;
    println!("{}", output.decode_output);

    Ok(())
}
```

## Serve API

```rust
use vllm_rs::api::{EngineBuilder, ModelRepo};

fn main() -> candle_core::Result<()> {
    let mut engine = EngineBuilder::new(ModelRepo::ModelID((
        "Qwen/Qwen3-0.6B".to_string(),
        None,
    )))
    .build()?;

    engine.start_server(8000, true, false)?;
    Ok(())
}
```

## Multi-rank / multi-GPU

Provide `device_ids` with `with_multirank` (e.g. `"0,1"`) along with the same CUDA/NCCL features
you use for the CLI. The Rust API reuses the same engine and scheduler path.

```rust
use vllm_rs::api::{EngineBuilder, ModelRepo};

fn main() -> candle_core::Result<()> {
    let mut engine = EngineBuilder::new(ModelRepo::ModelFile(vec![
        "/path/Qwen3-VL-8B-Instruct-GGUF-Q4_KM.gguf".to_string(),
    ]))
    .with_multirank("0,1")?
    .build()?;

    engine.start_server(8000, true, true)?;
    Ok(())
}
```
## Command to run

[Reference Rust demo](/example/rust-demo/)

```shell
# add `nccl` feature for multirank inference (and copy `runner` (which is built with build.sh) to your target path)
cargo run --release --features cuda
```