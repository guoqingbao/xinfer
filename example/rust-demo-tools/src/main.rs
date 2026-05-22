//! Rust Tool Calling Example for xInfer
//!
//! This example demonstrates:
//! 1. Defining tools using the builder pattern
//! 2. Parsing tool calls from model output
//! 3. Handling tool results

use xinfer::tools::parser::ToolParser;
use xinfer::tools::schema::SchemaBuilder;
use xinfer::tools::{Tool, ToolCall, ToolFormat, ToolResult};

fn main() -> anyhow::Result<()> {
    println!("xInfer Tool Calling Demo (Rust API)\n");

    // === Part 1: Define Tools ===
    println!("=== Part 1: Defining Tools ===\n");

    // Using builder pattern
    let weather_tool = Tool::function("get_weather", "Get the current weather for a location")
        .param("location", "string", "The city name", true)
        .param(
            "unit",
            "string",
            "Temperature unit (celsius/fahrenheit)",
            false,
        )
        .build();

    // Using schema builder for more complex schemas
    let search_schema = SchemaBuilder::object()
        .description("Web search parameters")
        .string_prop("query", "Search query", true)
        .integer_prop("max_results", "Maximum results", false)
        .build();

    let search_tool = Tool::function("search_web", "Search the web for information")
        .parameters_schema(search_schema)
        .build();

    println!(
        "Weather Tool: {}",
        serde_json::to_string_pretty(&weather_tool)?
    );
    println!(
        "\nSearch Tool: {}",
        serde_json::to_string_pretty(&search_tool)?
    );

    // === Part 2: Format Tools for Prompts ===
    println!("\n=== Part 2: Tool Prompt Formatting ===\n");

    let tools = vec![weather_tool.clone(), search_tool.clone()];

    let qwen_prompt = ToolFormat::format_tools(&tools);
    println!("Qwen Format:\n{}\n", qwen_prompt);

    // === Part 3: Parse Tool Calls from Model Output ===
    println!("=== Part 3: Parsing Tool Calls ===\n");

    let parser = ToolParser::new();

    // Simulate model output with tool call
    let model_outputs = vec![
        // Qwen format
        r#"I'll check the weather for you.
<tool_call>
{"name": "get_weather", "arguments": {"location": "Tokyo", "unit": "celsius"}}
</tool_call>"#,
        // JSON format
        r#"Let me search for that. {"name": "search_web", "arguments": {"query": "Rust programming"}}"#,
        // Code block format
        r#"Here's my search:
```json
{"name": "get_weather", "arguments": {"location": "London"}}
```"#,
    ];

    for (i, output) in model_outputs.iter().enumerate() {
        println!("Model Output {}:", i + 1);
        println!("  Input: {:?}", output.lines().next().unwrap_or(""));

        let calls = parser.parse(output);
        if calls.is_empty() {
            println!("  No tool calls detected\n");
        } else {
            for call in &calls {
                println!(
                    "  Parsed: {} with {}",
                    call.function.name, call.function.arguments
                );
            }
            println!();
        }
    }

    // === Part 4: Handle Tool Results ===
    println!("=== Part 4: Tool Results ===\n");

    // Simulate executing a tool and creating result
    let call = ToolCall::new("call_001", "get_weather", r#"{"location": "Tokyo"}"#);

    // Execute tool (simulated)
    let weather_data = serde_json::json!({
        "location": "Tokyo",
        "temperature": 22,
        "unit": "celsius",
        "condition": "sunny"
    });

    // Create result
    let result = ToolResult::success(&call.id, weather_data.to_string());
    println!("Tool Call ID: {}", call.id);
    println!("Tool Result: {}", result.content);

    // Error example
    let error_result = ToolResult::error("call_002", "API rate limit exceeded");
    println!(
        "\nError Result: {} (is_error: {:?})",
        error_result.content, error_result.is_error
    );

    println!("\n✅ Demo complete!");

    Ok(())
}
