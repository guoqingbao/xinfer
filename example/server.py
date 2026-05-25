import argparse
import multiprocessing as mp
import os
import signal
import sys
import warnings
from xinfer import Engine, EngineConfig, GenerationConfig, PdConfig, PdMethod, PdRole

def parse_args():
    parser = argparse.ArgumentParser(description="Run Chat Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--m", help="huggingface model id", type=str, default=None)
    parser.add_argument("--w", help="safetensor weight path", type=str, default=None)
    parser.add_argument("--f", help="gguf file path or gguf file name when model_id is given", type=str, default=None)
    parser.add_argument("--dtype", choices=["f16", "bf16", "f32"], default="bf16")
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--d", type=str, default="0")
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
    parser.add_argument("--pd-server", action="store_true")
    parser.add_argument("--pd-client", action="store_true")
    parser.add_argument("--pd-url", help="Url like `192.168.1.100:8888` \
        used for TCP/IP communication between PD server and client", type=str, default=None)
    parser.add_argument("--ui-server", action="store_true")
    parser.add_argument("--mcp_config", type=str, default=None)
    parser.add_argument("--mcp_command", type=str, default=None)
    parser.add_argument("--mcp_args", type=str, default=None)
    parser.add_argument("--enforce-parser", type=str, default=None)
    parser.add_argument("--pd-server-prefix-cache-ratio", type=float, default=None)
    parser.add_argument("--pd-client-prefix-cache-ratio", type=float, default=None)
    parser.add_argument("--disable-reasoning", action="store_true")
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--prefill-chunk-size", type=int, default=8192)

    args = parser.parse_args()
    if args.pd_server and args.ui_server:
        raise ValueError("PD Server cannot run with UI Server enabled!")
    if args.enforce_parser is not None:
        enforce_parser = args.enforce_parser.strip()
        if enforce_parser == "":
            args.enforce_parser = None
        else:
            valid_parsers = {
                "passthrough",
                "json",
                "mistral",
                "qwen",
                "qwen_coder",
                "pythonic",
                "llama",
                "deepseek",
                "glm45_moe",
                "glm47_moe",
                "step3",
                "kimik2",
                "minimax_m2",
            }
            if enforce_parser not in valid_parsers:
                valid_list = ", ".join(sorted(valid_parsers))
                raise ValueError(
                    f"Invalid --enforce-parser '{enforce_parser}'. Valid parsers: {valid_list}"
                )
            args.enforce_parser = enforce_parser
    return args

def run_server(args):
    # Build and run the engine in a child process so the parent can handle Ctrl+C.
    max_num_seqs = 1 if sys.platform == "darwin" else args.max_num_seqs

    generation_cfg = None
    if (args.temperature != None and (args.top_p != None or args.top_k != None)) or args.frequency_penalty != None or args.presence_penalty != None:
         generation_cfg = GenerationConfig(args.temperature, args.top_p, args.top_k, args.frequency_penalty, args.presence_penalty)

    assert args.m or args.w or args.f, "Must provide model_id or weight_path or weight_file!"
    if args.max_model_len != None:
        args.max_tokens = args.max_model_len if args.max_tokens > args.max_model_len else args.max_tokens
        
    assert args.max_model_len == None or args.kv_fraction == None, "You provided both max_model_len and kv_fraction!"

    pd_config = None
    if args.pd_server or args.pd_client:
        pd_role = PdRole.Server if args.pd_server else PdRole.Client
        # RemoteTcp for http:// or tcp:// URLs, LocalIpc for file:// or None
        if args.pd_url and (args.pd_url.startswith("tcp:") or args.pd_url.startswith("http:")):
            pd_method = PdMethod.RemoteTcp
        else:
            pd_method = PdMethod.LocalIpc
        pd_config = PdConfig(role=pd_role, method=pd_method, url=args.pd_url)

    cfg = EngineConfig(
        model_id=args.m,
        weight_path=args.w,
        weight_file=args.f,
        enforce_parser=args.enforce_parser,
        max_num_seqs=max_num_seqs,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        isq=args.isq,
        device_ids=[int(d) for d in args.d.split(",")],
        generation_cfg=generation_cfg,
        disable_prefix_cache=args.disable_prefix_cache,
        prefix_cache_max_tokens=args.prefix_cache_max_tokens,
        kvcache_dtype=args.kvcache_dtype,
        server_mode=True,
        cpu_mem_fold=args.cpu_mem_fold,
        kv_fraction=args.kv_fraction,
        pd_config=pd_config,
        mcp_config=args.mcp_config,
        mcp_command=args.mcp_command,
        mcp_args=args.mcp_args,
        pd_server_prefix_cache_ratio=args.pd_server_prefix_cache_ratio,
        pd_client_prefix_cache_ratio=args.pd_client_prefix_cache_ratio,
        disable_reasoning=args.disable_reasoning,
        disable_cuda_graph=args.disable_cuda_graph,
        prefill_chunk_size=args.prefill_chunk_size,
    )

    engine = Engine(cfg, args.dtype)

    # max_kvcache_tokens = max_model_len * max_num_seqs
    # if args.max_model_len is None:
    #     warnings.warn(f"Warning: max_model_len is not given, default to {max_model_len}, max kvcache tokens {max_kvcache_tokens}.")
    
    port = args.port if args.port is not None else (7000 if args.pd_server else 8000)
    engine.start_server(port, args.ui_server) # this will block


def main():
    args = parse_args()
    ctx = mp.get_context("spawn")
    server_proc = ctx.Process(target=run_server, args=(args,))
    server_proc.start()

    def _shutdown(signum, frame):
        if server_proc.is_alive():
            try:
                os.kill(server_proc.pid, signal.SIGINT)
            except OSError:
                pass
            server_proc.join(timeout=2.0)
            if server_proc.is_alive():
                server_proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        while server_proc.is_alive():
            server_proc.join(timeout=0.5)
    except KeyboardInterrupt:
        _shutdown(signal.SIGINT, None)

    sys.exit(server_proc.exitcode or 0)


if __name__ == "__main__":
    main()
