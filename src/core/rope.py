from typing import Dict, Optional, Union
import mlx.core as mx
import mlx.nn as nn

from .config import ModelArgs

class DynamicNTKScalingRoPE(nn.Module):
    """Rotary Position Embeddings with dynamic NTK scaling."""
    
    def __init__(
        self,
        dims: int,
        max_position_embeddings: int = 2048,
        base: float = 10000,
        scale: float = 1.0,
        traditional: bool = False,
        rope_type: str = "default",
        rope_scaling: Optional[Dict[str, Union[float, str]]] = None,
    ):
        super().__init__()
        self.dims = dims
        self.max_position_embeddings = max_position_embeddings
        self.original_base = base
        self.base = base
        self.scale = scale
        self.traditional = traditional
        self.rope_type = rope_type
        self.rope_scaling = rope_scaling

    def compute_ntk_scaling(self, seq_len: int) -> float:
        """Compute NTK scaling factor based on sequence length."""
        if seq_len <= self.max_position_embeddings:
            return 1.0
        return float(seq_len / self.max_position_embeddings)

    def compute_dynamic_scaling(self, seq_len: int) -> float:
        """Compute dynamic scaling factor."""
        if seq_len <= self.max_position_embeddings:
            return 1.0
        return float(mx.log(seq_len) / mx.log(self.max_position_embeddings))

    def compute_llama3_base_freq(self) -> float:
        """Compute base frequency for LLaMA3-style scaling."""
        if not self.rope_scaling:
            return self.original_base
            
        factor = self.rope_scaling.get("factor", 1.0)
        low_rank = self.rope_scaling.get("low_rank", 0.25)
        high_rank = self.rope_scaling.get("high_rank", 0.75)
        
        # Compute wavelengths
        wavelens = 1.0 / (self.original_base ** (mx.arange(self.dims) / self.dims))
        low_freq_wavelen = wavelens[int(self.dims * low_rank)]
        high_freq_wavelen = wavelens[int(self.dims * high_rank)]
        
        # Apply smooth scaling
        smooths = (wavelens - high_freq_wavelen) / (low_freq_wavelen - high_freq_wavelen)
        smooths = mx.clip(smooths, 0, 1)
        
        # Compute scaled frequencies
        new_base_freqs = wavelens * (1 - smooths) * factor + smooths
        new_base_freqs = mx.where(wavelens < high_freq_wavelen, wavelens, new_base_freqs)
        new_base_freqs = mx.where(wavelens > low_freq_wavelen, wavelens * factor, new_base_freqs)
        
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



def initialize_rope(args: ModelArgs) -> DynamicNTKScalingRoPE:
    """Initialize RoPE layer from model arguments."""
    head_dim = args.head_dim or args.hidden_size // args.num_attention_heads

    rope_scaling = args.rope_scaling
    rope_type = "default"
    rope_scale = 1.0

    if rope_scaling is not None:
        rope_type = rope_scaling.get("type") or rope_scaling.get("rope_type") or "default"
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