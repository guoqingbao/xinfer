from dataclasses import dataclass
from typing import Iterator, Tuple, Union, List, Literal, Mapping, Optional, Callable
from enum import Enum
from collections.abc import AsyncGenerator
from typing import Any

@dataclass
class DType(Enum):
    F16 = "f16"
    BF16 = "bf16"
    F32 = "f32"

@dataclass 
class PdRole(Enum):
    Client = 1
    PDServer = 2

@dataclass 
class PdMethod(Enum):
    LocalIpc = 1
    RemoteTcp = 2

@dataclass 
class PdConfig:
    role: PdRole
    method = PdMethod
    url: Optional[str]


@dataclass
class GenerationOutput:
    seq_id: int
    prompt_length: int
    prompt_start_time: int
    decode_start_time: int
    decode_finish_time: int
    decoded_length: int
    decode_output: str

@dataclass
class GenerationConfig:
    temperature: Optional[float]
    top_p: Optional[float]
    top_k: Optional[int]
    frequency_penalty: Optional[float]
    presence_penalty: Optional[float]

@dataclass
class EngineConfig:
    model_id: Optional[str]
    weight_path: Optional[str]
    weight_file: Optional[str]
    hf_token: Optional[str]
    hf_token_path: Optional[str]
    enforce_parser: Optional[str]
    max_num_seqs: Optional[int]
    config_model_len: Optional[int]
    max_model_len: Optional[int]
    max_tokens: Optional[int]
    isq: Optional[str]
    num_shards: Optional[int]
    device_ids: Optional[List[int]]
    generation_cfg: Optional[GenerationConfig]
    seed: Optional[int]
    disable_prefix_cache: bool
    prefix_cache_max_tokens: Optional[int]
    kvcache_dtype: Optional[str]
    server_mode: Optional[bool]
    cpu_mem_fold: Optional[float]
    kv_fraction: Optional[float]
    mamba_fraction: Optional[float]
    pd_config: Optional[PdConfig]
    mcp_command: Optional[str]
    mcp_config: Optional[str]
    mcp_args: Optional[List[str]]
    tool_prompt_template: Optional[str]
    pd_server_prefix_cache_ratio: Optional[float]
    pd_client_prefix_cache_ratio: Optional[float]
    yarn_scaling_factor: Optional[float]
    disable_reasoning: bool
    disable_cuda_graph: bool
    prefill_chunk_size: Optional[int]

    def __init__(
        self,
        model_id: Optional[str] = None,
        weight_path: Optional[str] = None,
        weight_file: Optional[str] = None,
        hf_token: Optional[str] = None,
        hf_token_path: Optional[str] = None,
        enforce_parser: Optional[str] = None,
        max_num_seqs: Optional[int] = 32,
        config_model_len: Optional[int] = None,
        max_model_len: Optional[int] = 1024,
        max_tokens: Optional[int] = None,
        isq: Optional[str] = None,
        num_shards: Optional[int] = 1,
        device_ids: Optional[List[int]] = None,
        generation_cfg: Optional[GenerationConfig] = None,
        seed: Optional[int] = None,
        disable_prefix_cache: bool = False,
        prefix_cache_max_tokens: Optional[int] = None,
        kvcache_dtype: Optional[str] = None,
        server_mode: Optional[bool] = None,
        cpu_mem_fold: Optional[float] = None,
        kv_fraction: Optional[float] = None,
        mamba_fraction: Optional[float] = None,
        pd_config: Optional[PdConfig] = None,
        mcp_command: Optional[str] = None,
        mcp_config: Optional[str] = None,
        mcp_args: Optional[List[str]] = None,
        tool_prompt_template: Optional[str] = None,
        pd_server_prefix_cache_ratio: Optional[float] = None,
        pd_client_prefix_cache_ratio: Optional[float] = None,
        yarn_scaling_factor: Optional[float] = None,
        disable_reasoning: bool = False,
        disable_cuda_graph: bool = False,
        prefill_chunk_size: Optional[int] = 8192,
    ) -> None: ...

@dataclass
class SamplingParams:
    temperature: Optional[float]
    max_tokens: Optional[int]
    ignore_eos: Optional[bool]
    top_k: Optional[int]
    top_p: Optional[float]
    session_id: Optional[str]
    frequency_penalty: Optional[float]
    presence_penalty: Optional[float]

@dataclass
class Message:
    role: str
    content: str
    num_images: int = 0

@dataclass
class StepOutput(Enum):
    Token: int
    Tokens: List[int]

@dataclass
class StreamItem:
    """
    An item returned by the EngineStream iterator.
    Check the `type` attribute to determine how to interpret the `data`.
    """
    @property
    def datatype(self) -> Literal["TOKEN", "TOKEN_ID", "COMPLETION", "DONE", "ERROR"]:
        """The type of the stream item."""
        ...

    @property
    def data(self) -> Union[
        str,                         # For TOKEN or ERROR
        int,                         # For TOKEN_ID
        Tuple[int, int, int, int],   # For DONE
        Tuple[int, int, int, List[int]] # For COMPLETION
    ]:
        """The data payload of the stream item."""
        ...

class EngineStream:
    finished: bool
    seq_id: int
    prompt_length: int
    cancelled: bool
    def cancel(self): ...
    def __iter__(self) -> Iterator[StreamItem]: ...
    def __next__(self) -> StreamItem: ...

class Engine:
    def __init__(econfig: EngineConfig, dtype: DType) -> Engine:
        """
        Create an xInfer engine with given engine config and dtype ("f16", "bf16", and "f32")
        """

    def generate_sync(self,
        params: List[SamplingParams],
        message_list: List[List[Message]],
    ) -> List[GenerationOutput]:
        """
        Chat completion using given prompts and sampling parameters
        """
    def generate_stream(
        self,
        params: SamplingParams,
        messages: List[Message],
    ) -> Tuple[int, int, EngineStream]:
        """
        Chat streaming using given prompts and sampling parameters.

        Return: (seq_id, prompt_length, stream) tuples
        """

    def get_num_cached_tokens(
        self,
    ) -> int:
        """
        Call this function when prefix-cache feature enabled

        Return: total number of cached prefix tokens
        """

    def get_available_kv_tokens(
        self,
    ) -> int:
        """
        Return: total number of available kvcache tokens
        """

    def start_server(
        self,
        port: int,
        with_ui_server: bool,
    ):
        """
        Start the API server with optional start of Chat UI server
        """
