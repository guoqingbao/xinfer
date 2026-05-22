#!/usr/bin/env python3
"""
Tokenize/Detokenize API Demo for xInfer

This script demonstrates how to use the /tokenize and /detokenize API endpoints.
Make sure the xInfer server is running before executing this script.

Usage:
    python tokenize.py [--url URL]
"""

import argparse
import requests
import json


def tokenize_text(base_url: str, text: str, add_special_tokens: bool = True) -> dict:
    """
    Tokenize plain text using the /tokenize endpoint.
    
    Args:
        base_url: Base URL of the xInfer server
        text: Text to tokenize
        add_special_tokens: Whether to add special tokens (default: True)
    
    Returns:
        API response containing tokens, count, and max_model_len
    """
    response = requests.post(
        f"{base_url}/tokenize",
        json={
            "prompt": text,
            "add_special_tokens": add_special_tokens
        }
    )
    response.raise_for_status()
    return response.json()


def tokenize_messages(base_url: str, messages: list, add_special_tokens: bool = True) -> dict:
    """
    Tokenize chat messages using the /tokenize endpoint.
    This applies the model's chat template before tokenization.
    
    Args:
        base_url: Base URL of the xInfer server
        messages: List of message dicts with 'role' and 'content' keys
        add_special_tokens: Whether to add special tokens (default: True)
    
    Returns:
        API response containing tokens, count, and max_model_len
    """
    response = requests.post(
        f"{base_url}/tokenize",
        json={
            "messages": messages,
            "add_special_tokens": add_special_tokens
        }
    )
    response.raise_for_status()
    return response.json()


def detokenize(base_url: str, tokens: list, skip_special_tokens: bool = True) -> dict:
    """
    Convert token IDs back to text using the /detokenize endpoint.
    
    Args:
        base_url: Base URL of the xInfer server
        tokens: List of token IDs to decode
        skip_special_tokens: Whether to skip special tokens in output (default: True)
    
    Returns:
        API response containing the decoded prompt
    """
    response = requests.post(
        f"{base_url}/detokenize",
        json={
            "tokens": tokens,
            "skip_special_tokens": skip_special_tokens
        }
    )
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser(description="Tokenize/Detokenize API Demo")
    parser.add_argument("--url", default="http://localhost:8000", help="xInfer server URL")
    args = parser.parse_args()
    
    base_url = args.url
    print(f"Using server at: {base_url}\n")
    
    # Example 1: Tokenize plain text
    print("=" * 50)
    print("Example 1: Tokenize plain text")
    print("=" * 50)
    text = "Hello, world! How are you today?"
    try:
        result = tokenize_text(base_url, text)
        print(f"Input: {text}")
        print(f"Tokens: {result['tokens']}")
        print(f"Token count: {result['count']}")
        if result.get('max_model_len'):
            print(f"Max model length: {result['max_model_len']}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        print("Make sure the xInfer server is running!")
        return
    
    print()
    
    # Example 2: Tokenize chat messages (applies chat template)
    print("=" * 50)
    print("Example 2: Tokenize chat messages")
    print("=" * 50)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2?"},
    ]
    try:
        result = tokenize_messages(base_url, messages)
        print(f"Messages: {json.dumps(messages, indent=2)}")
        print(f"Token count (with chat template): {result['count']}")
        print(f"First 10 tokens: {result['tokens'][:10]}...")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
    
    print()
    
    # Example 3: Detokenize
    print("=" * 50)
    print("Example 3: Detokenize tokens")
    print("=" * 50)
    # Use the tokens from Example 1
    try:
        tokens = tokenize_text(base_url, "Hello!")['tokens']
        print(f"Input tokens: {tokens}")
        result = detokenize(base_url, tokens)
        print(f"Decoded text: {result['prompt']}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
    
    print()
    
    # Example 4: Round-trip test
    print("=" * 50)
    print("Example 4: Round-trip test")
    print("=" * 50)
    original = "The quick brown fox jumps over the lazy dog."
    try:
        tokenized = tokenize_text(base_url, original)
        detokenized = detokenize(base_url, tokenized['tokens'])
        print(f"Original: {original}")
        print(f"After round-trip: {detokenized['prompt']}")
        print(f"Match: {original == detokenized['prompt']}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
