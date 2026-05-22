//! Tokenize/Detokenize API Demo for xInfer
//!
//! This demonstrates how to use the /tokenize and /detokenize endpoints.
//! Make sure the xinfer server is running before executing this.
//!
//! Usage:
//!     cargo run [-- --url http://localhost:8000]

use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::env;

// === Request/Response Types ===

#[derive(Serialize)]
struct TokenizeTextRequest {
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    add_special_tokens: Option<bool>,
}

#[derive(Serialize)]
struct TokenizeMessagesRequest {
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    add_special_tokens: Option<bool>,
}

#[derive(Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize, Debug)]
struct TokenizeResponse {
    tokens: Vec<u32>,
    count: usize,
    #[serde(default)]
    max_model_len: Option<usize>,
}

#[derive(Serialize)]
struct DetokenizeRequest {
    tokens: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    skip_special_tokens: Option<bool>,
}

#[derive(Deserialize, Debug)]
struct DetokenizeResponse {
    prompt: String,
}

// === API Functions ===

fn tokenize_text(client: &Client, base_url: &str, text: &str) -> Result<TokenizeResponse, Box<dyn std::error::Error>> {
    let request = TokenizeTextRequest {
        prompt: text.to_string(),
        add_special_tokens: Some(true),
    };
    
    let response = client
        .post(format!("{}/tokenize", base_url))
        .json(&request)
        .send()?
        .json::<TokenizeResponse>()?;
    
    Ok(response)
}

fn tokenize_messages(client: &Client, base_url: &str, messages: Vec<Message>) -> Result<TokenizeResponse, Box<dyn std::error::Error>> {
    let request = TokenizeMessagesRequest {
        messages,
        add_special_tokens: Some(true),
    };
    
    let response = client
        .post(format!("{}/tokenize", base_url))
        .json(&request)
        .send()?
        .json::<TokenizeResponse>()?;
    
    Ok(response)
}

fn detokenize(client: &Client, base_url: &str, tokens: Vec<u32>) -> Result<DetokenizeResponse, Box<dyn std::error::Error>> {
    let request = DetokenizeRequest {
        tokens,
        skip_special_tokens: Some(true),
    };
    
    let response = client
        .post(format!("{}/detokenize", base_url))
        .json(&request)
        .send()?
        .json::<DetokenizeResponse>()?;
    
    Ok(response)
}

fn main() {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let base_url = if args.len() > 2 && args[1] == "--url" {
        args[2].clone()
    } else {
        "http://localhost:8000".to_string()
    };
    
    println!("Using server at: {}\n", base_url);
    
    let client = Client::new();
    
    // Example 1: Tokenize plain text
    println!("{}", "=".repeat(50));
    println!("Example 1: Tokenize plain text");
    println!("{}", "=".repeat(50));
    
    let text = "Hello, world! How are you today?";
    match tokenize_text(&client, &base_url, text) {
        Ok(result) => {
            println!("Input: {}", text);
            println!("Tokens: {:?}", result.tokens);
            println!("Token count: {}", result.count);
            if let Some(max_len) = result.max_model_len {
                println!("Max model length: {}", max_len);
            }
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            eprintln!("Make sure the xinfer server is running!");
            return;
        }
    }
    
    println!();
    
    // Example 2: Tokenize chat messages
    println!("{}", "=".repeat(50));
    println!("Example 2: Tokenize chat messages");
    println!("{}", "=".repeat(50));
    
    let messages = vec![
        Message {
            role: "system".to_string(),
            content: "You are a helpful assistant.".to_string(),
        },
        Message {
            role: "user".to_string(),
            content: "What is 2 + 2?".to_string(),
        },
    ];
    
    match tokenize_messages(&client, &base_url, messages) {
        Ok(result) => {
            println!("Token count (with chat template): {}", result.count);
            println!("First 10 tokens: {:?}...", &result.tokens[..result.tokens.len().min(10)]);
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }
    
    println!();
    
    // Example 3: Detokenize
    println!("{}", "=".repeat(50));
    println!("Example 3: Detokenize tokens");
    println!("{}", "=".repeat(50));
    
    if let Ok(tokenized) = tokenize_text(&client, &base_url, "Hello!") {
        println!("Input tokens: {:?}", tokenized.tokens);
        match detokenize(&client, &base_url, tokenized.tokens) {
            Ok(result) => {
                println!("Decoded text: {}", result.prompt);
            }
            Err(e) => {
                eprintln!("Error: {}", e);
            }
        }
    }
    
    println!();
    
    // Example 4: Round-trip test
    println!("{}", "=".repeat(50));
    println!("Example 4: Round-trip test");
    println!("{}", "=".repeat(50));
    
    let original = "The quick brown fox jumps over the lazy dog.";
    if let Ok(tokenized) = tokenize_text(&client, &base_url, original) {
        if let Ok(detokenized) = detokenize(&client, &base_url, tokenized.tokens) {
            println!("Original: {}", original);
            println!("After round-trip: {}", detokenized.prompt);
            println!("Match: {}", original == detokenized.prompt);
        }
    }
}
