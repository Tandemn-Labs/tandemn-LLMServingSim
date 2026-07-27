"""Preserve Qwen2's unsharded head width during single-GPU TP emulation."""

from __future__ import annotations

import os
from functools import wraps
from typing import Any

from torch import nn


_HEAD_DIM_ENV = "LLMSERVINGSIM_QWEN2_PROFILE_HEAD_DIM"


def install() -> None:
    """Patch vLLM's Qwen2 attention only when the profiler requests it.

    Qwen2 derives ``head_dim`` from ``hidden_size / num_heads`` rather than
    honoring the HF config field. The profiler shrinks ``num_heads`` to mimic
    a TP rank while intentionally retaining the full hidden input width, so
    that derivation is wrong unless we preserve the original head width here.
    """
    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm.model_executor.layers.attention import Attention, EncoderOnlyAttention
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.model_executor.models.qwen2 import Qwen2Attention
    from vllm.model_executor.models.utils import extract_layer_index
    from vllm.v1.attention.backend import AttentionType

    if getattr(Qwen2Attention, "_llmservingsim_profile_patch", False):
        return

    original_init = Qwen2Attention.__init__

    @wraps(original_init)
    def patched_init(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict[str, Any],
        max_position: int = 4096 * 32,
        cache_config=None,
        quant_config=None,
        prefix: str = "",
        attn_type: AttentionType = AttentionType.DECODER,
        dual_chunk_attention_config=None,
        qk_norm: bool = False,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        requested_head_dim = os.environ.get(_HEAD_DIM_ENV)
        if requested_head_dim is None:
            original_init(
                self, hidden_size, num_heads, num_kv_heads, rope_parameters,
                max_position, cache_config, quant_config, prefix, attn_type,
                dual_chunk_attention_config, qk_norm, rms_norm_eps,
            )
            return

        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = int(requested_head_dim)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = dual_chunk_attention_config
        self.qk_norm = qk_norm
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        attn_cls = (
            EncoderOnlyAttention
            if attn_type == AttentionType.ENCODER_ONLY
            else Attention
        )
        self.attn = attn_cls(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            attn_type=attn_type,
            prefix=f"{prefix}.attn",
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": dual_chunk_attention_config,
            }
            if dual_chunk_attention_config
            else {},
        )

    Qwen2Attention.__init__ = patched_init
    Qwen2Attention._llmservingsim_profile_patch = True
