import time
import os
import sys
import argparse
import warnings
from xinfer import EngineConfig, SamplingParams, Message, GenerationOutput, GenerationConfig, Engine
# Before running this code, first perform maturin build and then install the package in target/wheels


def current_millis():
    return int(time.time() * 1000)


def run(args):
    prompts = args.prompts
    if prompts == None:
        if args.batch > 1:
            prompts = ["Please talk about China in more details."] * args.batch
        else:
            prompts = ["How are you?", "How to make money?"]
            print("⛔️ No prompts found, use default ", prompts)
    else:
        prompts = prompts.split("|")
        if args.batch > 1:
            prompts = prompts[0] * args.batch

    if args.batch > 1:
        max_num_seqs = args.batch
    elif len(prompts) > 0:
        max_num_seqs = len(prompts)
    else:
        # limit default max_num_seqs to 8 on MacOs (due to limited gpu memory)
        max_num_seqs = 8 if sys.platform == "darwin" else args.max_num_seqs

    if args.max_model_len is None:
        max_model_len = 32768 // max_num_seqs
        warnings.warn(f"max_model_len is not given, default to {max_model_len}.")
    else:
        max_model_len = args.max_model_len

    generation_cfg = None
    if (args.temperature != None and (args.top_p != None or args.top_k != None)) or args.frequency_penalty != None or args.presence_penalty != None:
         generation_cfg = GenerationConfig(args.temperature, args.top_p, args.top_k, args.frequency_penalty, args.presence_penalty)

    assert args.m or args.w or args.f, "Must provide model_id or weight_path or weight_file!"
    cfg = EngineConfig(
        model_id=args.m,
        weight_path=args.w,
        weight_file=args.f,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        max_tokens=max_model_len if args.max_tokens > max_model_len else args.max_tokens,
        isq=args.isq,
        device_ids=[int(d) for d in args.d.split(",")],
        generation_cfg=generation_cfg,
        disable_prefix_cache=args.disable_prefix_cache,
        prefix_cache_max_tokens=args.prefix_cache_max_tokens,
        kvcache_dtype=args.kvcache_dtype,
        server_mode=False,
        cpu_mem_fold=args.cpu_mem_fold,
        kv_fraction=args.kv_fraction,
        disable_reasoning=args.disable_reasoning,
        disable_cuda_graph=args.disable_cuda_graph,
        prefill_chunk_size=args.prefill_chunk_size,
    )


    engine = Engine(cfg, "bf16")

    sampling_params = []
    params = SamplingParams()
    message_list = []
    for i in range(len(prompts)):
        message_list.append([Message("user", prompts[i])])
        sampling_params.append(params)

    print("Start inference with", len(prompts), "prompts")
    outputs: GenerationOutput = engine.generate_sync(sampling_params, message_list)
    outputs.sort(key=lambda o: o.seq_id)

    decode_time_taken = 0.0
    prompt_time_taken = 0.0
    total_decoded_tokens = 0
    total_prompt_tokens = 0

    for i, output in enumerate(outputs):
        if args.batch == 1:
            print(f"\n[Prompt {i + 1}]")
            print(f"Prompt: {prompts[i]}")
            print(f"Response: {output.decode_output}")

        total_prompt_tokens += output.prompt_length
        total_decoded_tokens += output.decoded_length

        prompt_latency = (output.decode_start_time - output.prompt_start_time) / 1000.0
        prompt_time_taken = max(prompt_time_taken, prompt_latency)

        decode_latency = (output.decode_finish_time - output.decode_start_time) / 1000.0
        decode_time_taken = max(decode_time_taken, decode_latency)

    print("\n--- Performance Metrics ---")
    print(
        f"⏱️ Prompt tokens: {total_prompt_tokens} in {prompt_time_taken:.2f}s "
        f"({total_prompt_tokens / max(prompt_time_taken, 0.001):.2f} tokens/s)"
    )
    print(
        f"⏱️ Decoded tokens: {total_decoded_tokens} in {decode_time_taken:.2f}s "
        f"({total_decoded_tokens / max(decode_time_taken, 0.001):.2f} tokens/s)"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="xInfer Python CLI")
    parser.add_argument("--m", help="huggingface model id", type=str, default=None)
    parser.add_argument("--w", help="safetensor weight path", type=str, default=None)
    parser.add_argument("--f", help="gguf file path or gguf file name when model_id is given", type=str, default=None)
    parser.add_argument("--prompts", type=str,
                        help="Use '|' to separate multiple prompts")
    parser.add_argument("--d", type=str, default="0")
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--isq", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--frequency-penalty", type=float, default=None)
    parser.add_argument("--presence-penalty", type=float, default=None)
    parser.add_argument("--disable-prefix-cache", action="store_true")
    parser.add_argument("--prefix-cache-max-tokens", type=int, default=None)
    parser.add_argument("--kvcache-dtype", type=str, default=None, help="KV cache quantization: fp8, turbo8, turbo4, turbo3")
    parser.add_argument("--cpu-mem-fold", type=float, default=None)
    parser.add_argument("--kv-fraction", type=float, default=None)
    parser.add_argument("--disable-reasoning", action="store_true")
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--prefill-chunk-size", type=int, default=8192)

    args = parser.parse_args()
    if not os.path.exists(args.w):
        print("⛔️ Model path is not provided (--w)!")
    else:
        run(args)
