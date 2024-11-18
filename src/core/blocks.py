# core/blocks.py

from typing import Optional, Tuple, Dict, Any
import mlx.core as mx
import mlx.nn as nn

from .model import ModelArgs
from .attention import Attention, create_attention_mask

class MLP(nn.Module):
    """Multi-layer perceptron with SwiGLU activation."""
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        hidden_dim = args.intermediate_size
        
        # Determine if we should use bias
        mlp_bias = getattr(args, "mlp_bias", False)

        # Linear projections
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=mlp_bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=mlp_bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=mlp_bias)
        
        # Neuron-wise scaling factors (for emphasis/ablation)
        self.neuron_scale = mx.ones(hidden_dim, dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with SwiGLU activation."""
        # Gate and up-project
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        # SwiGLU activation
        hidden = nn.silu(gate) * up
        
        # Apply neuron-wise scaling
        neuron_scale = self.neuron_scale[None, None, :]
        hidden = hidden * neuron_scale
        
        # Down-project
        return self.down_proj(hidden)

class TransformerBlock(nn.Module):
    """Transformer block with pre-normalization."""
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        
        # Normalization layers
        self.input_layernorm = nn.RMSNorm(
            args.hidden_size,
            eps=args.rms_norm_eps
        )
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size,
            eps=args.rms_norm_eps
        )
        
        # Attention and MLP
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        
        # Layer scaling factor (for emphasis/ablation)
        self.layer_scale = mx.array(1.0, dtype=mx.float32)
        
        # Store args for reference
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> Tuple[mx.array, mx.array, mx.array, Dict[str, Any], Optional[Any]]:
        """Forward pass for transformer block."""
        # Skip if layer is fully ablated
        if self.layer_scale == 0.0:
            return x, None, x, {}, cache
            
        # Pre-normalization and attention
        pre_attn = self.input_layernorm(x)
        attn_out, attn_weights, _, attn_data, new_cache = self.self_attn(
            pre_attn, mask, cache
        )
        
        # First residual connection with layer scaling
        h = x + attn_out * self.layer_scale
        
        # Pre-normalization and MLP
        pre_mlp = self.post_attention_layernorm(h)
        mlp_out = self.mlp(pre_mlp)
        
        # Second residual connection with layer scaling
        out = h + mlp_out * self.layer_scale
        
        return out, attn_weights, pre_attn, attn_data, new_cache

    def update_layer_scale(self, scale: float) -> None:
        """Update layer scaling factor."""
        self.layer_scale = mx.array(scale, dtype=mx.float32)

class DecoderLayer(TransformerBlock):
    """Alias for TransformerBlock to match common nomenclature."""
    pass

def get_activation_fn(activation: str):
    """Get activation function by name."""
    if activation == "gelu":
        return nn.gelu
    elif activation == "relu":
        return nn.relu
    elif activation == "silu":
        return nn.silu
    else:
        raise ValueError(f"Unsupported activation: {activation}")

class LayerNormFactory:
    """Factory for creating different types of layer normalization."""
    
    @staticmethod
    def create(
        norm_type: str,
        dim: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> nn.Module:
        if norm_type == "rms_norm":
            return nn.RMSNorm(dim, eps=eps)
        elif norm_type == "layer_norm":
            return nn.LayerNorm(dim, eps=eps)
        else:
            raise ValueError(f"Unsupported normalization: {norm_type}")

def create_block(
    block_type: str,
    args: ModelArgs,
    **kwargs
) -> nn.Module:
    """Factory function for creating different types of blocks."""
    if block_type == "transformer":
        return TransformerBlock(args, **kwargs)
    elif block_type == "mlp":
        return MLP(args, **kwargs)
    else:
        raise ValueError(f"Unsupported block type: {block_type}")