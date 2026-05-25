import time
import argparse
import warnings
import sys
import readline # input without cutoff
from xinfer import Engine, EngineConfig, SamplingParams, Message, GenerationConfig
# Before running this code, first perform maturin build and then install the package in target/wheels


def current_millis():
    return int(time.time() * 1000)


def parse_args():
    parser = argparse.ArgumentParser(description="xInfer Python CLI")
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--m", help="huggingface model id", type=str, default=None)
    parser.add_argument("--w", help="safetensor weight path", type=str, default=None)
    parser.add_argument("--f", help="gguf file path or gguf file name when model_id is given", type=str, default=None)
    parser.add_argument("--dtype", type=str,
                        choices=["f16", "bf16", "f32"], default="bf16")
    parser.add_argument("--d", type=str, default="0")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--prompts", type=str,
                        help="Use '|' to separate multiple prompts")
    parser.add_argument("--i", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--isq", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--presence-penalty", type=float, default=None)
    parser.add_argument("--frequency-penalty", type=float, default=None)
    parser.add_argument("--disable-prefix-cache", action="store_true")
    parser.add_argument("--prefix-cache-max-tokens", type=int, default=None)
    parser.add_argument("--kvcache-dtype", type=str, default=None, help="KV cache quantization: fp8, turbo8, turbo4, turbo3")
    parser.add_argument("--cpu-mem-fold", type=float, default=None)
    parser.add_argument("--kv-fraction", type=float, default=None)
    parser.add_argument("--disable-reasoning", action="store_true")
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--prefill-chunk-size", type=int, default=8192)

    return parser.parse_args()


def build_engine_config(args, num_of_prompts, disable_prefix_cache):
    generation_cfg = None
    if (args.temperature != None and (args.top_p != None or args.top_k != None)) or args.frequency_penalty != None or args.presence_penalty != None:
         generation_cfg = GenerationConfig(args.temperature, args.top_p, args.top_k, args.frequency_penalty, args.presence_penalty)

    assert args.m or args.w or args.f, "Must provide model_id or weight_path or weight_file!"
    if args.max_model_len != None:
        args.max_tokens=args.max_model_len if args.max_tokens > args.max_model_len else args.max_tokens
    
    assert args.max_model_len == None or args.kv_fraction == None, "You provided both max_model_len and kv_fraction!"
    
    return EngineConfig(
        model_id=args.m,
        weight_path=args.w,
        weight_file=args.f,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        isq=args.isq,
        device_ids=[int(d) for d in args.d.split(",")],
        generation_cfg=generation_cfg,
        disable_prefix_cache=disable_prefix_cache,
        prefix_cache_max_tokens=args.prefix_cache_max_tokens,
        kvcache_dtype=args.kvcache_dtype,
        server_mode=False,
        cpu_mem_fold=args.cpu_mem_fold,
        kv_fraction=args.kv_fraction,
        disable_reasoning=args.disable_reasoning,
        disable_cuda_graph=args.disable_cuda_graph,
        prefill_chunk_size=args.prefill_chunk_size,
    )

def show_tokens_left(tokens_left: int, total_tokens: int):
    import shutil
    width = shutil.get_terminal_size(fallback=(80, 20)).columns
    if tokens_left < 0: 
        tokens_left = 0
    token_info = f"Tokens left: {tokens_left}"

    # Choose color based on remaining tokens
    if tokens_left * 1.0 / total_tokens > 0.5:
        color = "\033[32m"  # Green
    elif tokens_left * 1.0 / total_tokens > 0.1:
        color = "\033[33m"  # Yellow
    else:
        color = "\033[31m"  # Red

    reset = "\033[0m"

    # Calculate padding
    space_padding = width - 2 - len(token_info)
    space_padding = max(1, space_padding)  # prevent negative spacing

    # Build the final line
    line = (" " * space_padding) + color + token_info + reset

    print(line)

def remove_surrogates(s: str) -> str:
    return ''.join(c for c in s if not (0xD800 <= ord(c) <= 0xDFFF))

def main():
    args = parse_args()
    interactive = args.i
    interactive = True # disable non-interactive mode for now
    disable_prefix_cache = args.disable_prefix_cache
    prompts = (
        args.prompts.split("|")
        if args.prompts and not interactive
        else ["How are you today?"]
    )

    econfig = build_engine_config(args, len(prompts), disable_prefix_cache)
    engine = Engine(econfig, args.dtype)

    if args.prompts and interactive:
        print("[Warning] Ignoring predefined prompts in interactive mode.")
        prompts = []

    sampling_params = []

    prompt_processed = []
    params = SamplingParams()
    message_list = []
    if not interactive:
        for prompt in prompts:
            msg = Message(role="user", content=remove_surrogates(prompt))
            message_list.append([msg])
            sampling_params.append(params)
    else:
        sampling_params.append(params)

    total_available_tokens = engine.get_available_kv_tokens()
    tokens_left = total_available_tokens
    chat_history = []
    while True:
        if interactive:
            try:
                show_tokens_left(tokens_left, total_available_tokens)
                prompt_input = input(
                    "\n🤖✨ Enter your prompt (or paste as one line, Ctrl+C to reset chat, Ctrl+D to exit):\n> ")
                if not prompt_input:
                    continue
                msg = Message(role="user", content=remove_surrogates(prompt_input))
                chat_history.append(msg)
                params.session_id = None
                # (prompt_processed, prompt_uuid) = [
                #     engine.apply_chat_template(params, chat_history, log=False)]

            except KeyboardInterrupt:
                if chat_history:
                    print("\n🌀 Chat history cleared. Start a new conversation.")
                    chat_history.clear()
                    if not disable_prefix_cache:
                        tokens_left = total_available_tokens - engine.get_num_cached_tokens()
                    else:
                        tokens_left = total_available_tokens
                    continue
                else:
                    print("\n👋 Exiting.")
                    break

            except EOFError:
                print("\n👋 Exiting.")
                break

        if interactive:
            decoded_length = 0
            decode_start_time = current_millis()
            try:
                done_item = None
                output_text = ""
                seq_id, prompt_length, stream = engine.generate_stream(params, chat_history)
                for item in stream:
                    if item.datatype == "TOKEN":
                        print(item.data[0], end="", flush=True)
                        output_text += item.data[0]
                        if decoded_length == 0:
                            decode_start_time = current_millis()
                        decoded_length += 1
                    elif item.datatype == "ERROR":
                        raise Exception(item.data[0])
                    elif item.datatype == "DONE":
                        done_item = item.data

                print()  # newline after streaming ends
                if done_item != None:
                    prompt_start_time, decode_start_time, decode_finish_time, decoded_length = done_item
                if not disable_prefix_cache:
                    tokens_left = total_available_tokens - engine.get_num_cached_tokens()
                else:
                    tokens_left = total_available_tokens - prompt_length - decoded_length
                # Construct a GenerationOutput-like object manually
                if done_item != None:
                    output = type("GenerationOutput", (), {
                        "seq_id": seq_id,
                        "decode_output": output_text,
                        "prompt_length": prompt_length,
                        "prompt_start_time": prompt_start_time,
                        "decode_start_time": decode_start_time,
                        "decode_finish_time": decode_finish_time,
                        "decoded_length": decoded_length,
                    })()
                    outputs = [output]
                else:
                    outputs = []
            except KeyboardInterrupt:
                stream.cancel()
                if not disable_prefix_cache:
                    tokens_left = total_available_tokens - engine.get_num_cached_tokens()
                else:
                    tokens_left = total_available_tokens
                print("\n⛔️ Interrupted by user!")
                if decoded_length > 0:
                    print("\n⏱️ [Unfinished] Decode throughput: ", round(decoded_length * 1000 / (current_millis() - decode_start_time), 2), " tokens/s")
                continue
            except Exception as e:
                chat_history.clear()
                if not disable_prefix_cache:
                    tokens_left = total_available_tokens - engine.get_num_cached_tokens()
                else:
                    tokens_left = total_available_tokens
                print("\n⛔️", e, ", chat session closed!")
                continue
        else:
            outputs = engine.generate_sync(sampling_params, message_list)

        if len(outputs) > 0:
            outputs.sort(key=lambda o: o.seq_id)

        decode_time_taken = 0.0
        prompt_time_taken = 0.0
        total_decoded_tokens = 0
        total_prompt_tokens = 0

        for i, output in enumerate(outputs):
            if not interactive and len(prompts) > 1:
                print(f"\n[Prompt {i + 1}]")
                print(f"Prompt: {prompts[i]}")
                print(f"Response: {output.decode_output}")

            total_prompt_tokens += output.prompt_length
            total_decoded_tokens += output.decoded_length

            prompt_latency = (output.decode_start_time - output.prompt_start_time) / 1000.0
            prompt_time_taken = max(prompt_time_taken, prompt_latency)

            decode_latency = (output.decode_finish_time -
                              output.decode_start_time) / 1000.0
            decode_time_taken = max(decode_time_taken, decode_latency)

            if interactive:
                assistant_msg = Message(
                    role="assistant", content=output.decode_output)
                chat_history.append(assistant_msg)

        color = "\033[37m"
        if len(outputs) > 0:
            print(color + "\n--- Performance Metrics ---")
            print(
                color + f"⏱️ Prompt tokens: {total_prompt_tokens} in {prompt_time_taken:.2f}s "
                f"({total_prompt_tokens / max(prompt_time_taken, 0.001):.2f} tokens/s)"
            )
            print(
                color + f"⏱️ Decoded tokens: {total_decoded_tokens} in {decode_time_taken:.2f}s "
                f"({total_decoded_tokens / max(decode_time_taken, 0.001):.2f} tokens/s)"
            )

        if not interactive:
            break


if __name__ == "__main__":
    main()
