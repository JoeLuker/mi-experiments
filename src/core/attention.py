# core/attention.py

from typing import Optional, Dict, Union
import inspect

import mlx.core as mx
import mlx.nn as nn

from .model import ModelArgs

class DynamicNTKScalingRoPE(nn.Module):
    """Enhanced rotary position encoding with multiple scaling strategies."""

    SCALING_TYPES = ["default", "dynamic", "linear", "ntk", "llama3"]
    
    def __init__(
        self,
        dims: int,
        max_position_embeddings: int = 2048,
        traditional: bool = False,
        base: float = 10000,
        scale: float = 1.0,
        rope_type: str = "default",
        rope_scaling: Optional[Dict[str, Union[float, str]]] = None,
    ):
        super().__init__()
        if rope_type not in self.SCALING_TYPES:
            raise ValueError(f"Unsupported scaling type: {rope_type}. Must be one of {self.SCALING_TYPES}")
            
        self.dims = dims
        self.max_position_embeddings = max_position_embeddings
        self.traditional = traditional
        self.original_base = base
        self.scale = scale
        self.rope_type = rope_type
        self.rope_scaling = rope_scaling
        self.base = self.compute_base_freq()

    def compute_ntk_scaling(self, seq_len: int) -> float:
        """Compute NTK-aware scaling factor."""
        if seq_len <= self.max_position_embeddings:
            return 1.0
        
        return (
            (self.scale * seq_len / self.max_position_embeddings) - (self.scale - 1)
        ) ** (self.dims / (self.dims - 2))

    def compute_dynamic_scaling(self, seq_len: int) -> float:
        """Compute dynamic scaling factor based on sequence length."""
        if not self.rope_scaling or "factor" not in self.rope_scaling:
            return 1.0
            
        factor = self.rope_scaling["factor"]
        alpha = self.rope_scaling.get("alpha", 1.0)
        target_len = self.max_position_embeddings * factor
        
        if seq_len <= self.max_position_embeddings:
            return 1.0
            
        return (target_len / seq_len) ** alpha

    def compute_llama3_base_freq(self) -> float:
        """Enhanced LLaMA 3 RoPE scaling."""
        if not self.rope_scaling:
            return self.original_base
            
        factor = self.rope_scaling["factor"]
        low_freq_factor = self.rope_scaling.get("low_freq_factor", 1.0)
        high_freq_factor = self.rope_scaling.get("high_freq_factor", 4.0)
        old_context_len = self.rope_scaling.get(
            "original_max_position_embeddings",
            8192,
        )

        # Compute frequency bands
        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        # Create frequency tensor
        freqs = self.original_base ** (mx.arange(0, self.dims, 2) / self.dims)
        wavelens = 2 * mx.pi * freqs

        # Apply smooth scaling
        smooths = (wavelens - high_freq_wavelen) / (
            low_freq_wavelen - high_freq_wavelen
        )
        smooths = mx.clip(smooths, 0, 1)

        # Compute scaled frequencies
        new_base_freqs = freqs * (1 - smooths) * factor + smooths
        new_base_freqs = mx.where(wavelens < high_freq_wavelen, freqs, new_base_freqs)
        new_base_freqs = mx.where(
            wavelens > low_freq_wavelen, freqs * factor, new_base_freqs
        )

        return new_base_freqs.mean().item()

    def compute_base_freq(self) -> float:
        """Compute base frequency based on scaling type."""
        if self.rope_type == "llama3":
            return self.compute_llama3_base_freq()
        elif self.rope_type == "linear" and self.rope_scaling:
            return self.original_base / self.rope_scaling["factor"]
        return self.original_base

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        """Apply rotary position encoding with appropriate scaling."""
        seq_len = x.shape[1] + offset
        base = self.base
        
        # Apply scaling based on type
        if self.rope_type == "ntk":
            scale_factor = self.compute_ntk_scaling(seq_len)
            base *= scale_factor
        elif self.rope_type == "dynamic":
            scale_factor = self.compute_dynamic_scaling(seq_len)
            base *= scale_factor

        return mx.fast.rope(
            x,
            self.dims,
            traditional=self.traditional,
            base=base,
            scale=self.scale,
            offset=offset,
        )

def create_additive_causal_mask(N: int, offset: int = 0) -> mx.array:
    """Create causal attention mask."""
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    mask = linds[:, None] < rinds[None]
    return mask * -1e9

def create_attention_mask(h: mx.array, cache: Optional[Any] = None) -> Optional[mx.array]:
    """Create attention mask for transformer."""
    T = h.shape[1]
    if T > 1:
        if cache is not None and cache[0] is not None:
            offset = cache[0].offset
        else:
            offset = 0
        mask = create_additive_causal_mask(T, offset)
        return mask.astype(h.dtype)
    return None

def initialize_rope(args: ModelArgs) -> DynamicNTKScalingRoPE:
    """Initialize RoPE layer from model arguments."""
    head_dim = args.head_dim or args.hidden_size // args.num_attention_heads

    rope_scaling = args.rope_scaling
    rope_type = "default"
    rope_scale = 1.0

    if rope_scaling is not None:
        rope_type = (
            rope_scaling.get("type") or rope_scaling.get("rope_type") or "default"
        )
        if rope_type == "linear":
            rope_scale = 1 / rope_scaling["factor"]
        elif rope_type == "llama3":
            rope_scale = 1.0  # Scaling handled internally for llama3

    return DynamicNTKScalingRoPE(
        dims=head_dim,
        max_position_embeddings=args.max_position_embeddings,
        traditional=args.rope_traditional,
        base=args.rope_theta,
        scale=rope_scale,
        rope_type=rope_type,
        rope_scaling=rope_scaling,
    )

class ScaledQueryAttention(nn.Module):
    """Optimized attention implementation with query scaling."""
    
    def __init__(
        self,
        query_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = query_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(query_dim, query_dim, bias=bias)
        self.k_proj = nn.Linear(query_dim, query_dim, bias=bias)
        self.v_proj = nn.Linear(query_dim, query_dim, bias=bias)
        self.out_proj = nn.Linear(query_dim, query_dim, bias=bias)

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass for scaled query attention."""
        batch_size = query.shape[0]
        
        # Project queries, keys, values
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        # Reshape for attention
        query = query.reshape(batch_size, -1, self.num_heads, self.head_dim)
        key = key.reshape(batch_size, -1, self.num_heads, self.head_dim)
        value = value.reshape(batch_size, -1, self.num_heads, self.head_dim)

        # Scale query
        query = query * self.scaling

        # Compute attention scores
        attn = mx.matmul(query, key.transpose(0, 1, 3, 2))
        
        if mask is not None:
            attn = attn + mask

        attn = mx.softmax(attn, axis=-1)

        # Apply attention to values
        output = mx.matmul(attn, value)
        
        # Reshape and project output
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.num_heads * self.head_dim)
        return self.out_proj(output)