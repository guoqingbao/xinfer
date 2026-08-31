# 推测解码（MTP 与 DFlash2）

xInfer 通过**推测解码**加速自回归生成：每步先由草稿模型提出若干候选 token，再由目标模型在一次前向中批量验证；接受的草稿可直接推进序列，减少逐 token 解码轮次。

| 模式 | 适用场景 | CLI |
|------|----------|-----|
| **内置 MTP** | 目标模型自带 MTP 预测头（如 Qwen3.5 / Qwen3.6） | `--num-speculative-tokens N` |
| **DFlash2** | 独立 DFlash2 草稿模型（单独 safetensors 权重） | `--draft-model <id或路径>` |

**规则：** 设置 `--draft-model` 时启用 **DFlash2**（不支持 DFlash1 草稿）；否则在目标模型含 MTP 层时，`--num-speculative-tokens` 启用**内置 MTP**。

---

## 内置 MTP

MTP（Multi-Token Prediction）使用目标权重中的轻量预测头。每步解码流程：

1. 目标模型生成 **anchor** token 与隐藏状态；
2. MTP 头自回归草稿 `N` 个 token（不额外增长 KV）；
3. 目标模型对 `[anchor, draft₀, …, draftₙ₋₁]` 做一次验证前向；
4. 接受匹配前缀；Qwen3.5 混合架构在部分拒绝时自动回滚 GDN/Mamba 状态。

### 命令行示例

```bash
# Qwen3.5 35B，每步 3 个推测 token
xinfer --m Qwen/Qwen3.5-35B-A3B --d 0,1 \
  --num-speculative-tokens 3 --ui-server
```

建议草稿数：**3–7**。过大时接受率波动与 KV 验证宽度都会增加。

### 要求

- 目标权重需包含 MTP 层（`mtp.*` 或 GGUF `nextn.*`）。
- 当前支持：**Qwen3.5**、**Qwen3.5 MoE**、**Qwen3-VL**（Qwen3.5 文本骨干）。
- 推荐编译：`./build.sh --release --features cuda,nccl,flashinfer,cutlass`

---

## DFlash2

[DFlash2](https://github.com/z-lab/dflash) 使用**独立草稿模型**，消费目标网络选定层的中间隐状态。草稿 token 通过 top-k 候选格点（`selector_top_k`）选择。

### 命令行示例

```bash
# Qwen3.8 目标 + DFlash2 草稿（HuggingFace ID 或本地路径）
xinfer --m Qwen/Qwen3.8-... --d 0,1 \
  --draft-model <dflash2-草稿仓库或路径> \
  --num-speculative-tokens 7 --ui-server
```

`--draft-model` 可为 **HuggingFace 模型 ID** 或含 `config.json` 与 safetensors 的**本地目录**。

`--num-speculative-tokens` 为每步草稿 token 数（不含 anchor）。省略时从草稿配置读取（`block_size - 1`）。

草稿权重须为 **DFlash2** safetensors（`architectures` 含 `DFlash2` 或 `dflash_config.selector_top_k` 已设置）。不支持 DFlash1 与 GGUF 草稿。

### 要求

- 目标：**Qwen3.5**、**Qwen3.5 MoE**、**Qwen3-VL** 或 **Qwen3.8**（dense）。
- 草稿：同系列 **DFlash2** 权重。
- 推荐编译：`cuda,nccl,flashinfer,cutlass`。

---

## Python API

```python
from xinfer import Engine, EngineConfig

# 内置 MTP
cfg = EngineConfig(
    model_id="Qwen/Qwen3.5-35B-A3B",
    num_speculative_tokens=3,
)

# DFlash2（draft_model 启用 DFlash2，禁用内置 MTP）
cfg = EngineConfig(
    model_id="Qwen/Qwen3.8-...",
    draft_model="<dflash2-草稿-id或路径>",
    num_speculative_tokens=7,
)
```

---

## 参数说明

| 参数 | 说明 |
|------|------|
| `--num-speculative-tokens N` | 每步草稿 token 数。未设置 `--draft-model` 且在含 MTP 的模型上启用 **MTP**；与 `--draft-model` 联用时指定 DFlash2 宽度（可省略，默认读草稿配置）。 |
| `--draft-model <id或路径>` | 外部 **DFlash2** 草稿模型，启用 DFlash2 并替代内置 MTP。 |

**已移除：** `--mtp`（请用 `--num-speculative-tokens`）、`--draft-model-id` / `--draft-model-path`（请用 `--draft-model`）。

---

## 注意事项

- **吞吐：** 加速取决于草稿接受率；日志中有 `MTP Stats` / `DFlash2 Stats` 汇总。
- **显存：** 验证阶段每步最多追加 `N+1` 个 token，注意 `--kv-fraction` 与 `--max-num-seqs`。
- **批处理：** 推测解码主要针对 `batch_size=1` 解码步优化。
- **PD 分离：** 混合 Mamba 模型请勿与 PD 模式同时使用推测解码。

英文详细文档：[speculative_decoding.md](./speculative_decoding.md)
