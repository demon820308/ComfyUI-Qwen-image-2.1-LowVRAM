import asyncio
import importlib.util
from pathlib import Path
import sys

import torch

COMFY_ROOT = Path(__file__).resolve().parents[3]
NODE_DIR = Path(__file__).resolve().parents[1]
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

import comfy.model_patcher
import comfy.ops
from comfy.ldm.qwen_image21.model import (
    QwenImage21Transformer2DModel,
    block_causal_attention,
    prefix_cached_attention,
)


def _load_node_module():
    pkg_name = "comfyui_qwen21_lowvram_testpkg"
    init_spec = importlib.util.spec_from_file_location(
        pkg_name,
        NODE_DIR / "__init__.py",
        submodule_search_locations=[str(NODE_DIR)],
    )
    pkg = importlib.util.module_from_spec(init_spec)
    sys.modules[pkg_name] = pkg
    init_spec.loader.exec_module(pkg)
    return sys.modules[f"{pkg_name}.nodes"]


node_mod = _load_node_module()
QwenImage21LowVRAM = node_mod.QwenImage21LowVRAM
comfy_entrypoint = node_mod.comfy_entrypoint


class _FakeModelWrapper(torch.nn.Module):
    def __init__(self, diffusion_model):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.load_device = torch.device("cpu")
        self.offload_device = torch.device("cpu")
        self.current_weight_patches_uuid = None

    def model_size(self):
        return 0


class _DummyCache:
    def __init__(self):
        self.stored = {}

    def put(self, block_index, tensor):
        self.stored[block_index] = tensor


def test_entrypoint_and_validation():
    ext = comfy_entrypoint()
    nodes = asyncio.run(ext.get_node_list())
    assert nodes == [QwenImage21LowVRAM]
    schema = QwenImage21LowVRAM.define_schema()
    assert schema.node_id == "QwenImage21LowVRAM"

    bad_patcher = comfy.model_patcher.ModelPatcher(
        _FakeModelWrapper(torch.nn.Linear(4, 4)),
        load_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
    )
    try:
        QwenImage21LowVRAM.execute(bad_patcher)
        raise AssertionError("Expected ValueError for non-Qwen2.1 model")
    except ValueError:
        pass


def _init_weights(model):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 1 or "norm" in name:
                p.fill_(1.0)
            else:
                torch.nn.init.normal_(p, mean=0.0, std=0.02)


def test_model_patcher_and_equivalence():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    torch.manual_seed(0)
    ops = comfy.ops.manual_cast
    model = QwenImage21Transformer2DModel(
        in_channels=64,
        out_channels=64,
        num_layers=2,
        attention_head_dim=32,
        num_attention_heads=8,
        context_in_dim=64,
        mlp_ratio=3,
        axes_dims_rope=(4, 14, 14),
        fused_mlp=True,
        dtype=dtype,
        device=device,
        operations=ops,
    ).eval()
    _init_weights(model)

    x = torch.randn(1, 64, 8, 8, device=device, dtype=dtype)
    t = torch.tensor([0.5], device=device, dtype=dtype)
    ctx = torch.randn(1, 16, 64, device=device, dtype=dtype)

    with torch.no_grad():
        ref_out = model(x, t, ctx)

    patcher = comfy.model_patcher.ModelPatcher(
        _FakeModelWrapper(model),
        load_device=device,
        offload_device=torch.device("cpu"),
    )
    orig_opts = dict(patcher.model_options.get("transformer_options", {}))

    out_node = QwenImage21LowVRAM.execute(
        patcher,
        ffn_chunks=4,
        head_chunks=4,
        seq_threshold=16,
        prefix_cache_mode="off",
        cleanup_memory=True,
    )
    patched = out_node[0]

    assert patcher.model_options.get("transformer_options", {}) == orig_opts
    assert len(patcher.object_patches) == 0
    assert patched.model_options["transformer_options"]["qwen_image21_cache"] == {"device": "off", "dtype": "default"}

    patched.patch_model(load_weights=False)
    try:
        with torch.no_grad():
            patched_out = model(x, t, ctx, transformer_options=patched.model_options["transformer_options"])
    finally:
        patched.unpatch_model(unpatch_weights=False)

    torch.testing.assert_close(patched_out, ref_out, atol=2e-2, rtol=2e-2)


def test_attention_paths_and_non_fused_ffn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    torch.manual_seed(1)
    ops = comfy.ops.manual_cast
    model = QwenImage21Transformer2DModel(
        in_channels=64,
        out_channels=64,
        num_layers=1,
        attention_head_dim=32,
        num_attention_heads=8,
        context_in_dim=64,
        mlp_ratio=2,
        axes_dims_rope=(4, 14, 14),
        fused_mlp=False,
        dtype=dtype,
        device=device,
        operations=ops,
    ).eval()
    _init_weights(model)

    block = model.transformer_blocks[0]
    x_mlp = torch.randn(1, 64, block.img_mlp.out.out_features, device=device, dtype=dtype)
    with torch.no_grad():
        ref_mlp = block.img_mlp(x_mlp)

    patcher = comfy.model_patcher.ModelPatcher(
        _FakeModelWrapper(model),
        load_device=device,
        offload_device=torch.device("cpu"),
    )
    patched = QwenImage21LowVRAM.execute(
        patcher,
        ffn_chunks=3,
        head_chunks=3,
        seq_threshold=16,
        prefix_cache_mode="cpu_int8",
        cleanup_memory=False,
    )[0]
    assert patched.model_options["transformer_options"]["qwen_image21_cache"] == {"device": "cpu", "dtype": "int8"}

    x_attn = torch.randn(1, 48, block.attn.to_q.in_features, device=device, dtype=dtype)
    ids = torch.arange(48, device=device, dtype=torch.float32).unsqueeze(1).expand(48, 3).unsqueeze(0)
    pe = model.pe_embedder(ids).transpose(1, 2).contiguous()
    segments = [(0, 16, torch.ones((16, 16), dtype=torch.bool, device=device).tril()), (16, 48, None)]
    prefix_k = torch.randn(1, 16, 8, 32, device=device, dtype=dtype)
    prefix_v = torch.randn(1, 16, 8, 32, device=device, dtype=dtype)

    with torch.no_grad():
        ref_causal = block.attn(x_attn, pe, block_causal_attention(segments, {}, cache=None, block_index=0, prefix_len=16), 16)
        ref_cached = block.attn(x_attn, pe, prefix_cached_attention(prefix_k, prefix_v, {}), 0)

    patched.patch_model(load_weights=False)
    try:
        with torch.no_grad():
            got_mlp = block.img_mlp(x_mlp)
            got_causal = block.attn(x_attn, pe, block_causal_attention(segments, {}, cache=None, block_index=0, prefix_len=16), 16)
            got_cached = block.attn(x_attn, pe, prefix_cached_attention(prefix_k, prefix_v, {}), 0)
            dummy_cache = _DummyCache()
            block.attn(x_attn, pe, block_causal_attention(segments, {}, cache=dummy_cache, block_index=0, prefix_len=16), 16)
            assert dummy_cache.stored[0].shape == (1, 2, 16, 8, 32)
    finally:
        patched.unpatch_model(unpatch_weights=False)

    torch.testing.assert_close(got_mlp, ref_mlp, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(got_causal, ref_causal, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(got_cached, ref_cached, atol=2e-2, rtol=2e-2)


def benchmark_2k_activation_vram():
    if not torch.cuda.is_available():
        print("CUDA unavailable; skipping 2K VRAM benchmark.")
        return

    from comfy.ldm.qwen_image21.model import SwiGLUFeedForward, Attention

    device = torch.device("cuda")
    dtype = torch.bfloat16
    ops = comfy.ops.manual_cast
    dim, hidden_dim, heads, head_dim = 4096, 12288, 32, 128
    seq_len = 16384  # 2048x2048 latent = 128x128 tokens

    mlp = SwiGLUFeedForward(dim, hidden_dim, fused=True, dtype=dtype, device=device, operations=ops).eval()
    attn = Attention(dim, heads, head_dim, dtype=dtype, device=device, operations=ops).eval()
    _init_weights(mlp)
    _init_weights(attn)
    x = torch.randn(1, seq_len, dim, device=device, dtype=dtype)
    pe = torch.zeros(1, seq_len, 1, head_dim // 2, 2, 2, device=device, dtype=torch.float32)
    pe[..., 0, 0] = 1.0
    pe[..., 1, 1] = 1.0
    segments = [(0, seq_len, None)]

    def measure(fn):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        with torch.no_grad():
            _ = fn()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base
        return peak / (1024 * 1024)

    base_mlp_mb = measure(lambda: mlp(x))
    mlp.forward = node_mod._MethodPatch(node_mod._chunked_swiglu_forward, _qwen21_ffn_chunks=4, _qwen21_seq_threshold=2048).__get__(mlp, mlp.__class__)
    chunk_mlp_mb = measure(lambda: mlp(x))

    base_attn_mb = measure(lambda: attn(x, pe, block_causal_attention(segments), 0))
    attn.forward = node_mod._MethodPatch(node_mod._lowvram_attn_forward, _qwen21_head_chunks=4, _qwen21_seq_threshold=2048).__get__(attn, attn.__class__)
    chunk_attn_mb = measure(lambda: attn(x, pe, block_causal_attention(segments), 0))

    print(f"2K (N=16384) FFN peak delta:  {base_mlp_mb:.1f} MiB -> {chunk_mlp_mb:.1f} MiB")
    print(f"2K (N=16384) Attn peak delta: {base_attn_mb:.1f} MiB -> {chunk_attn_mb:.1f} MiB")
    assert chunk_mlp_mb < base_mlp_mb


if __name__ == "__main__":
    test_entrypoint_and_validation()
    test_model_patcher_and_equivalence()
    test_attention_paths_and_non_fused_ffn()
    benchmark_2k_activation_vram()
    print("ALL TESTS PASSED")
