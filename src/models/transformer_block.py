from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import mlx.core as mx
import mlx.nn as nn
from .attention import Attention
from .mlp import MLP
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        
        # Main components
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        
        # Layer norms
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        
        # Store args for later reference
        self.args = args
        
        # Layer emphasis scaling (default to 1.0)
        self.layer_scale = mx.array(1.0, dtype=mx.float32)

    def forward(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
        return_attention_weights: bool = False
    ) -> Tuple[mx.array, Optional[mx.array]]:
        # Apply input layernorm
        normed_hidden_states = self.input_layernorm(hidden_states)
        
        # Self attention
        if return_attention_weights:
            attention_output, attention_weights = self.self_attn(
                normed_hidden_states,
                attention_mask,
                cache=cache,
                return_attention_weights=True
            )
        else:
            attention_output = self.self_attn(
                normed_hidden_states,
                attention_mask,
                cache=cache
            )
            attention_weights = None
        
        # Apply layer emphasis scaling
        attention_output = attention_output * self.layer_scale
        
        # Residual connection
        hidden_states = hidden_states + attention_output
        
        # MLP
        normed_hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(normed_hidden_states)
        
        # Apply layer emphasis scaling
        mlp_output = mlp_output * self.layer_scale
        
        # Final residual connection
        hidden_states = hidden_states + mlp_output
        
        if return_attention_weights:
            return hidden_states, attention_weights
        return hidden_states

    def update_emphasis(self, layer_scale: Optional[float] = None):
        """Update emphasis values for this transformer block"""
        if layer_scale is not None:
            self.layer_scale = mx.array(layer_scale, dtype=mx.float32)
            logger.debug(f"Updated layer scale to {layer_scale}") 