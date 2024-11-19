# core/attention.py

from typing import Optional, Dict, Union, Any, Tuple
import inspect

import mlx.core as mx
import mlx.nn as nn

from .config import ModelArgs
from .rope import initialize_rope, DynamicNTKScalingRoPE

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads or n_heads
        self.head_dim = head_dim = args.head_dim or dim // n_heads
        self.scale = head_dim**-0.5
        
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=args.attention_bias)
        
        self.rope = initialize_rope(args)
        self.head_scale = mx.ones((max(n_heads, n_kv_heads),), dtype=mx.float32)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape
        head_dim = D // self.n_heads

        # Project to queries, keys, values
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        # Reshape to [batch_size, seq_len, n_heads, head_dim]
        queries = queries.reshape(B, L, self.n_heads, head_dim)
        keys = keys.reshape(B, L, self.n_kv_heads, head_dim)
        values = values.reshape(B, L, self.n_kv_heads, head_dim)

        # Transpose to [batch_size, n_heads, seq_len, head_dim]
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        # Handle KV cache if provided
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # Compute attention scores
        scale = 1.0 / mx.sqrt(head_dim)
        scores = mx.matmul(queries, keys.transpose(0, 1, 3, 2)) * scale

        # Apply mask if provided - reshape mask for broadcasting
        if mask is not None:
            # Reshape mask to [batch_size, 1, 1, seq_length] for broadcasting
            mask = mask[:, None, None, :]
            scores = scores + mask

        # Compute attention weights and output
        weights = mx.softmax(scores, axis=-1)
        output = mx.matmul(weights, values)

        # Reshape back to [batch_size, seq_len, hidden_size]
        output = output.transpose(0, 2, 1, 3).reshape(B, L, D)
        
        return self.o_proj(output)

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