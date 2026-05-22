use xinfer::api::{EngineBuilder, ModelRepo};
use xinfer::server::{ChatMessage, MessageContentType};
use xinfer::utils::{config::SamplingParams, log_throughput};

fn main() -> anyhow::Result<()> {
    let mut engine = EngineBuilder::new(ModelRepo::ModelID(("Qwen/Qwen3-0.6B", None))).build()?;

    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: MessageContentType::PureText("Say hello from the Rust API.".to_string()),
    }];

    let params = SamplingParams::default();
    let output = engine.generate(params, messages)?;
    println!("\n\n{}", output.decode_output);

    log_throughput(&vec![output]);
}
