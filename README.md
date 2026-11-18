# ComfyUI-Qwen-image-2.1-LowVRAM

适用于 **ComfyUI 原生 Qwen-Image-2.1** 的低显存（8–10GB VRAM）与低内存（16GB RAM）优化节点，用于在保持原工作流与输出等价性的前提下运行更高分辨率（如 2K / 4K）生成与多图编辑。

> **技术边界说明**：本节点通过算子激活分块、Prefix KV Cache 管控与采样前后显存/内存回收降低峰值占用，**不修改模型权重精度**，可与 ComfyUI 原生 `aimdo` 动态加载、`BlockCache-T8`、`SageAttention`、`Comfy Kitchen` 共存。

## 理论与上游参考（Papers & Upstream Foundations）

本节点的设计针对 **Qwen-Image-2.1** 的架构瓶颈，结合了以下论文与上游工程思路：

1. **Qwen-Image 架构报告** — *Qwen-Image Technical Report* ([arXiv:2508.02324](https://arxiv.org/abs/2508.02324))
   - 上游采用大型多模态模型作为条件编码器（2.1 使用 `Qwen3-VL 8B`）+ DiT 主干（32 层、`inner_dim=4096`、`mlp_ratio=3`）+ 高压缩 VAE。高分辨率（2K/4K）与多图编辑会显著拉长联合序列长度，放大单层瞬时激活显存与条件缓存占用。
2. **DiT 激活峰值与旋转量化研究** — *ViDiT-Q* / *DiRotQ* / *SVDQuant*
   - 相关研究指出 DiT 推理时的显存峰值主要来自 **SwiGLU FFN 中间膨胀张量**（Qwen-Image-2.1 中 `gate_up` 将维度从 `4096` 膨胀 6 倍至 `24576`）及 **QKV/注意力投影缓存**；同时 ConvRot/per-token 激活量化具有行独立性。
   - **本节点对应实现**：`ffn_chunks` 沿 token 维度切分 `SwiGLUFeedForward`（实测 2K 单层 FFN 峰值增量由 `1280.0 MiB` 降至 `448.0 MiB`），`head_chunks` 沿注意力头分组执行 Attention 并在投影后立即释放归一化输入 `x`，数学上与未分块等价。
3. **DiT 分层卸载与异步预取** — *ChunkFlow*（Layerwise Offloading & Prefetching）
   - ComfyUI 原生已通过 `aimdo` + `model_prefetch` 对 `QwenImage21` 实现逐 Block 异步预取。本节点压低单层激活峰值后，可为 `aimdo` 腾出更多显存常驻权重块，减少 8–10GB 显卡上的 CPU-GPU 频繁换页。
4. **Diffusion KV Cache 复用与量化** — *dKV-Cache* / *KV Cache Quantization for Diffusion Models*
   - Qwen-Image-2.1 原生支持跨去噪步复用文本与参考图的 Prefix KV Cache（对应 ComfyUI `QwenImage21Cache`）。在 **16GB 系统内存** 环境下，8.7GB 文本编码器与 6.8GB DiT 已接近内存上限，本节点提供 `prefix_cache_mode`（默认 `off`，可选 `cpu_int8` / `cpu_int4` / `keep`）避免缓存与模型争抢内存导致 Swap 卡顿或 OOM。
5. **社区工程参考**
   - 参考了 ComfyUI 社区 `comfyui-kjnodes`（`MiniMaxChunkFeedForward`、`MiniMaxLowVRAMAttention`、`Ideogram4OptimizationsKJ`）中的 `add_object_patch` 算子分块与即时张量释放模式。

## 安装 / Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/demon820308/ComfyUI-Qwen-image-2.1-LowVRAM.git
```

重启 ComfyUI 后，在画布搜索 **`Qwen Image 2.1 Low VRAM`**（节点 ID：`QwenImage21LowVRAM`，分类：`model/conditioning/qwen image`）。

## 工作流接法

```text
UNETLoader (或 LoRA / Attention 后端) → Qwen Image 2.1 Low VRAM → KSampler → VAEDecodeTiled
```

- **8–10GB 显存 + 16GB 内存推荐参数**：
  - `ffn_chunks`：`4`（2K 推荐 `4`，4K 推荐 `8`–`16`）
  - `head_chunks`：`4`（4K 可调到 `8`）
  - `seq_threshold`：`2048`
  - `prefix_cache_mode`：`off`（16GB 内存推荐关闭，避免与文本编码器争抢内存；内存充裕可选 `cpu_int8` / `cpu_int4`）
  - `cleanup_memory`：`True`（采样前后自动 GC 与清理 CUDA 缓存）
- **高分辨率解码提示**：2K / 4K 出图请将工作流最后的 `VAEDecode` 替换为 ComfyUI 自带的 **`VAEDecodeTiled`（VAE解码（分块））**，默认 `tile_size=512, overlap=64` 即可避免最后一步解码 OOM。

## 测试 / Tests

```bash
python tests/test_qwen21_lowvram.py
```
