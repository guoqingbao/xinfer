#!/usr/bin/env python3
"""
Tool Calling Example for xInfer

This example demonstrates how to use tool calling with the xInfer server.
It shows:
1. Defining tools with JSON Schema
2. Sending chat completion requests with tools
3. Handling tool calls from the model
4. Sending tool results back to continue the conversation
"""

import requests
import json
import argparse

# Server URL
BASE_URL = "http://localhost:8000/v1"


def define_tools():
    """Define available tools with JSON Schema."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name, e.g., 'Tokyo', 'New York'"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit"
                        }
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Evaluate a mathematical expression",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The mathematical expression to evaluate, e.g., '2 + 2'"
                        }
                    },
                    "required": ["expression"]
                }
            }
        },
        {
            "type": "function", 
            "function": {
                "name": "search_web",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]


def execute_tool(tool_name: str, arguments: dict) -> str:
    """Execute a tool and return the result."""
    if tool_name == "get_weather":
        location = arguments.get("location", "Unknown")
        unit = arguments.get("unit", "celsius")
        # Simulated weather data
        temp = 22 if unit == "celsius" else 72
        return json.dumps({
            "location": location,
            "temperature": temp,
            "unit": unit,
            "condition": "sunny",
            "humidity": 65
        })
    
    elif tool_name == "calculator":
        expression = arguments.get("expression", "0")
        try:
            # WARNING: In production, use a safe expression parser!
            result = eval(expression)
            return json.dumps({"result": result})
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    elif tool_name == "search_web":
        query = arguments.get("query", "")
        # Simulated search results
        return json.dumps({
            "query": query,
            "results": [
                {"title": f"Result 1 for '{query}'", "url": "https://example.com/1"},
                {"title": f"Result 2 for '{query}'", "url": "https://example.com/2"}
            ]
        })
    
    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})


def chat_with_tools(user_message: str, tools: list, model: str = "default"):
    """Send a chat request with tools and handle tool calls."""
    
    messages = [{"role": "user", "content": user_message}]
    
    print(f"📝 User: {user_message}")
    print("-" * 50)
    
    while True:
        # Make the API request
        response = requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "tools": tools,
                "max_tokens": 2048
            }
        )
        
        if response.status_code != 200:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return
        
        result = response.json()
        choice = result["choices"][0]
        message = choice["message"]
        finish_reason = choice.get("finish_reason", "unknown")
        
        # Check if the model made tool calls
        if message.get("tool_calls"):
            print("🔧 Model is calling tools:")
            tool_results = []
            
            for tool_call in message["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                arguments = json.loads(tool_call["function"]["arguments"])
                tool_call_id = tool_call["id"]
                
                print(f"   - {tool_name}({json.dumps(arguments)})")
                
                # Execute the tool
                result = execute_tool(tool_name, arguments)
                print(f"     → Result: {result}")
                
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result
                })
            
            # Add the assistant's tool call message and tool results
            messages.append({
                "role": "assistant",
                "tool_calls": message["tool_calls"]
            })
            messages.extend(tool_results)
            
            print("-" * 50)
            continue
        
        # Model gave a text response
        content = message.get("content", "")
        print(f"🤖 Assistant: {content}")
        
        if finish_reason != "tool_calls":
            break
    
    return content


def main():
    parser = argparse.ArgumentParser(description="Tool calling example for xInfer")
    parser.add_argument("--url", default="http://localhost:8000/v1", 
                        help="Base URL for the xInfer server")
    parser.add_argument("--model", default="default", 
                        help="Model name to use")
    args = parser.parse_args()
    
    global BASE_URL
    BASE_URL = args.url
    
    tools = define_tools()
    
    print("=" * 60)
    print("🛠️  xInfer Tool Calling Demo")
    print("=" * 60)
    print()
    
    # Example 1: Weather query
    print("Example 1: Weather Query")
    print("=" * 60)
    chat_with_tools(
        "What's the weather like in Tokyo and New York?",
        tools,
        args.model
    )
    print()
    
    # Example 2: Calculator
    print("Example 2: Calculator")
    print("=" * 60)
    chat_with_tools(
        "What is 25 * 17 + 43?",
        tools,
        args.model
    )
    print()
    
    # Example 3: Web search
    print("Example 3: Web Search")
    print("=" * 60)
    chat_with_tools(
        "Search for information about Rust programming language",
        tools,
        args.model
    )
    print()
    
    # Interactive mode
    print("=" * 60)
    print("💬 Interactive Mode (type 'quit' to exit)")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\n🤖✨ Enter your prompt: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            if not user_input:
                continue
            
            print()
            chat_with_tools(user_input, tools, args.model)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except EOFError:
            print("\n👋 Goodbye!")
            break


if __name__ == "__main__":
    main()
