import gc
import logging
import types

import torch
import torch.nn.functional as F
from comfy_api.latest import ComfyExtension, io

import comfy.model_management
import comfy.ops
import comfy.patcher_extension as patches
import comfy.quant_ops
from comfy.ldm.flux.math import apply_rope1
from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention
from comfy.ldm.qwen_image21.model import QwenImage21Transformer2DModel

WRAPPER_KEY = "qwen_image21_lowvram"
CACHE_MODE_MAP = {
    "off": {"device": "off", "dtype": "default"},
    "cpu_int8": {"device": "cpu", "dtype": "int8"},
    "cpu_int4": {"device": "cpu", "dtype": "int4"},
}


def _chunk_sizes(total, chunks):
    base, rem = divmod(total, chunks)
    return [base + (1 if i < rem else 0) for i in range(chunks) if base + (1 if i < rem else 0) > 0]


def _closure_cell(fn, name):
    code = getattr(fn, "__code__", None)
    closure = getattr(fn, "__closure__", None)
    if code is None or closure is None or name not in code.co_freevars:
        return None
    return closure[code.co_freevars.index(name)].cell_contents


def _chunked_swiglu_forward(self, x):
    chunks = getattr(self, "_qwen21_ffn_chunks", 1)
    seq_threshold = getattr(self, "_qwen21_seq_threshold", 2048)
    if chunks <= 1 or x.shape[1] <= seq_threshold:
        if self.fused:
            return comfy.ops.linear_input_act(self.out, self.gate_up(x), "swiglu")
        return self.out(F.silu(self.gate_layer(x)) * self.proj(x))

    out = torch.empty_like(x)
    offset = 0
    for c in torch.chunk(x, chunks, dim=1):
        n = c.shape[1]
        if self.fused:
            out[:, offset:offset + n] = comfy.ops.linear_input_act(self.out, self.gate_up(c), "swiglu")
        else:
            out[:, offset:offset + n] = self.out(F.silu(self.gate_layer(c)) * self.proj(c))
        offset += n
    return out


def _attn_head_chunked(self, qkv, attn_fn, seq_threshold, head_chunks):
    q, k, v = qkv
    qkv.clear()
    B, N, H, D = q.shape
    if head_chunks <= 1 or H <= 1:
        return attn_fn(AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v), H, preferred_attention=self.comfy_attention)

    # prefix_cached_attention captures (prefix_k, prefix_v, transformer_options)
    prefix_k = _closure_cell(attn_fn, "prefix_k")
    prefix_v = _closure_cell(attn_fn, "prefix_v")
    if prefix_k is not None and prefix_v is not None:
        if N + prefix_k.shape[1] <= seq_threshold:
            return attn_fn(AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v), H, preferred_attention=self.comfy_attention)
        t_opts = _closure_cell(attn_fn, "transformer_options") or {}
        out = torch.empty((B, N, H * D), dtype=q.dtype, device=q.device)
        hs = 0
        sizes = _chunk_sizes(H, head_chunks)
        for idx, size in enumerate(sizes):
            he = hs + size
            qh = AttentionTensorContainer(q[:, :, hs:he].flatten(2))
            kh = AttentionTensorContainer(torch.cat([prefix_k[:, :, hs:he], k[:, :, hs:he]], dim=1).flatten(2))
            vh = AttentionTensorContainer(torch.cat([prefix_v[:, :, hs:he], v[:, :, hs:he]], dim=1).flatten(2))
            if idx == len(sizes) - 1:
                del q, k, v
            out[:, :, hs * D:he * D] = optimized_attention(
                qh, kh, vh, size, transformer_options=t_opts, preferred_attention=self.comfy_attention
            )
            hs = he
        return out

    # block_causal_attention captures (segments, cache, ...); only slice when cache is None so step-0 cache.put gets all heads
    has_segments = _closure_cell(attn_fn, "segments") is not None
    cache = _closure_cell(attn_fn, "cache")
    if has_segments and cache is None and N > seq_threshold:
        out = torch.empty((B, N, H * D), dtype=q.dtype, device=q.device)
        hs = 0
        sizes = _chunk_sizes(H, head_chunks)
        for idx, size in enumerate(sizes):
            he = hs + size
            if idx == len(sizes) - 1:
                qh = AttentionTensorContainer(q[:, :, hs:he].contiguous())
                kh = AttentionTensorContainer(k[:, :, hs:he].contiguous())
                vh = AttentionTensorContainer(v[:, :, hs:he].contiguous())
                del q, k, v
            else:
                qh = AttentionTensorContainer(q[:, :, hs:he])
                kh = AttentionTensorContainer(k[:, :, hs:he])
                vh = AttentionTensorContainer(v[:, :, hs:he])
            out[:, :, hs * D:he * D] = attn_fn(qh, kh, vh, size, preferred_attention=self.comfy_attention)
            hs = he
        return out

    return attn_fn(AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v), H, preferred_attention=self.comfy_attention)


def _lowvram_attn_forward(self, x, pe, attn_fn, prefix_len, transformer_options={}):
    B, N, _ = x.shape
    q = self.to_q(x).view(B, N, self.heads, -1)
    k = self.to_k(x).view(B, N, self.heads, -1)
    v = self.to_v(x).view(B, N, self.heads, -1)
    # Free normed block input immediately before RoPE and attention workspace allocation.
    del x

    attn1_patches = transformer_options.get("patches", {}).get("attn1_patch", [])
    if comfy.model_management.in_training or attn1_patches:
        q, k = self.norm_q(q), self.norm_k(k)
        if attn1_patches:
            q, k, v, pe = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), pe.transpose(1, 2)
            extra_options = {**transformer_options, "img_slice": [prefix_len, N]}
            for p in attn1_patches:
                out = p(q, k, v, pe=pe, attn_mask=None, extra_options=extra_options)
                q, k, v, pe = out.get("q", q), out.get("k", k), out.get("v", v), out.get("pe", pe)
            q, k, v, pe = q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous(), pe.transpose(1, 2).contiguous()
        q = apply_rope1(q, pe)
        k = apply_rope1(k, pe)
    else:
        q_scale, _, q_stream = comfy.ops.cast_bias_weight(self.norm_q, q, offloadable=True)
        k_scale, _, k_stream = comfy.ops.cast_bias_weight(self.norm_k, k, offloadable=True)
        q, k = comfy.quant_ops.ck.rms_rope(q, k, pe, q_scale, k_scale, self.norm_q.eps)
        comfy.ops.uncast_bias_weight(self.norm_q, q_scale, None, q_stream)
        comfy.ops.uncast_bias_weight(self.norm_k, k_scale, None, k_stream)

    head_chunks = getattr(self, "_qwen21_head_chunks", 1)
    seq_threshold = getattr(self, "_qwen21_seq_threshold", 2048)
    qkv = [q, k, v]
    del q, k, v
    out = _attn_head_chunked(self, qkv, attn_fn, seq_threshold, head_chunks)
    return self.to_out[0](out)


class _MethodPatch:
    def __init__(self, fn, **attrs):
        self.fn = fn
        self.attrs = attrs

    def __get__(self, obj, objtype=None):
        def wrapped(self_module, *args, **kwargs):
            for k, v in self.attrs.items():
                setattr(self_module, k, v)
            return self.fn(self_module, *args, **kwargs)
        return types.MethodType(wrapped, obj)


def _lowvram_sample_wrapper(executor, *args, **kwargs):
    gc.collect()
    comfy.model_management.soft_empty_cache()
    try:
        return executor(*args, **kwargs)
    finally:
        gc.collect()
        comfy.model_management.soft_empty_cache()


class QwenImage21LowVRAM(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="QwenImage21LowVRAM",
            display_name="Qwen Image 2.1 Low VRAM",
            category="model/conditioning/qwen image",
            description=(
                "Reduces peak VRAM and 16GB system RAM pressure for Qwen-Image-2.1 at high resolutions (e.g. 2K) "
                "via token-chunked SwiGLU FFN, head-chunked Attention, and optional Prefix KV Cache offloading."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Int.Input(
                    "ffn_chunks", default=4, min=1, max=32, step=1,
                    tooltip="Split SwiGLU FFN over tokens. Higher values cut peak MLP VRAM (4-8 recommended for 2K on 8-10GB GPUs).",
                ),
                io.Int.Input(
                    "head_chunks", default=4, min=1, max=32, step=1,
                    tooltip="Split Attention over head groups to reduce peak attention workspace VRAM.",
                ),
                io.Int.Input(
                    "seq_threshold", default=2048, min=256, max=65536, step=256,
                    tooltip="Only chunk when sequence length exceeds this token count (1024x1024 = 4096 tokens, 2048x2048 = 16384 tokens).",
                ),
                io.Combo.Input(
                    "prefix_cache_mode", options=["off", "cpu_int8", "cpu_int4", "keep"], default="off",
                    tooltip="off saves both VRAM and 16GB RAM (recommended for 16GB RAM). cpu_int8/cpu_int4 caches prefix K/V in RAM for faster edits when RAM allows. keep preserves upstream QwenImage21Cache.",
                ),
                io.Boolean.Input(
                    "cleanup_memory", default=True,
                    tooltip="Flush Python GC and CUDA allocator cache before and after sampling to leave headroom for CLIP and 2K VAE Decode.",
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, ffn_chunks=4, head_chunks=4, seq_threshold=2048, prefix_cache_mode="off", cleanup_memory=True) -> io.NodeOutput:
        diffusion_model = model.get_model_object("diffusion_model")
        if not isinstance(diffusion_model, QwenImage21Transformer2DModel):
            raise ValueError("QwenImage21LowVRAM requires ComfyUI's native Qwen-Image-2.1 model (QwenImage21Transformer2DModel).")
        m = model.clone()

        for idx, block in enumerate(diffusion_model.transformer_blocks):
            if ffn_chunks > 1:
                patched_mlp = _MethodPatch(_chunked_swiglu_forward, _qwen21_ffn_chunks=ffn_chunks, _qwen21_seq_threshold=seq_threshold).__get__(block.img_mlp, block.img_mlp.__class__)
                m.add_object_patch(f"diffusion_model.transformer_blocks.{idx}.img_mlp.forward", patched_mlp)
            patched_attn = _MethodPatch(_lowvram_attn_forward, _qwen21_head_chunks=head_chunks, _qwen21_seq_threshold=seq_threshold).__get__(block.attn, block.attn.__class__)
            m.add_object_patch(f"diffusion_model.transformer_blocks.{idx}.attn.forward", patched_attn)

        if prefix_cache_mode in CACHE_MODE_MAP:
            t_opts = dict(m.model_options.get("transformer_options", {}))
            t_opts["qwen_image21_cache"] = dict(CACHE_MODE_MAP[prefix_cache_mode])
            m.model_options["transformer_options"] = t_opts

        m.remove_wrappers_with_key(patches.WrappersMP.OUTER_SAMPLE, WRAPPER_KEY)
        if cleanup_memory:
            m.add_wrapper_with_key(patches.WrappersMP.OUTER_SAMPLE, WRAPPER_KEY, _lowvram_sample_wrapper)

        logging.info(
            "QwenImage21LowVRAM: patched %d blocks (ffn_chunks=%d, head_chunks=%d, seq_threshold=%d, prefix_cache=%s, cleanup=%s)",
            len(diffusion_model.transformer_blocks), ffn_chunks, head_chunks, seq_threshold, prefix_cache_mode, cleanup_memory,
        )
        return io.NodeOutput(m)


class Qwen21LowVRAMExtension(ComfyExtension):
    async def get_node_list(self):
        return [QwenImage21LowVRAM]


def comfy_entrypoint():
    return Qwen21LowVRAMExtension()
