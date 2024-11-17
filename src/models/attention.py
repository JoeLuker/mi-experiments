import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads or args.num_attention_heads
        self.head_dim = args.head_dim or args.hidden_size // args.num_attention_heads
        
        self.q_proj = nn.Linear(args.hidden_size, self.n_heads * self.head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(args.hidden_size, self.n_kv_heads * self.head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(args.hidden_size, self.n_kv_heads * self.head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, args.hidden_size, bias=args.attention_bias)
        
        self.scale = self.head_dim ** -0.5
        
        # Initialize head scaling factors (for emphasis/ablation)
        self.head_scale = mx.array([1.0] * self.n_heads, dtype=mx.float32)

    def forward(self, x: mx.array, mask: Optional[mx.array] = None, 
               return_attention_weights: bool = False) -> Union[mx.array, Tuple[mx.array, mx.array]]:
        B, L, _ = x.shape

        # Project to Q, K, V
        queries = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        # Apply head scaling (including ablation if scaling is zero)
        head_scale = self.head_scale[None, :, None, None]
        queries = queries * head_scale
        if self.n_heads == self.n_kv_heads:
            keys = keys * head_scale
            values = values * head_scale

        # Compute attention
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, 
            scale=self.scale,
            mask=mask
        )

        # Reshape and project output
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        output = self.o_proj(output)

        if return_attention_weights:
            if self.n_heads != self.n_kv_heads:
                keys = mx.repeat(keys, self.n_heads // self.n_kv_heads, axis=1)
            attention_weights = mx.softmax(
                (queries @ keys.transpose(0, 1, 3, 2)) * self.scale,
                axis=-1
            )
            return output, attention_weights
        return output
