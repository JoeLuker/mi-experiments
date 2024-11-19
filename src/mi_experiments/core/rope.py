import mlx.core as mx
import mlx.nn as nn

from mi_experiments.core.config import ModelArgs

class DynamicNTKScalingRoPE(nn.Module):
    """Implements the rotary positional encoding with Dynamic NTK scaling and Llama 3 RoPE."""

    def __init__(
        self,
        dims: int,
        max_position_embeddings: int = 2048,
        traditional: bool = False,
        base: float = 10000,
        scale: float = 1.0,
        rope_type: str = "default",
        rope_scaling: dict = None,
    ):
        super().__init__()
        self.dims = dims
        self.max_position_embeddings = max_position_embeddings
        self.traditional = traditional
        self.original_base = base
        self.scale = scale
        self.rope_type = rope_type
        self.rope_scaling = rope_scaling
        self.base = self.compute_base_freq()

    def compute_base_freq(self):
        if self.rope_type == "llama3":
            return self.compute_llama3_base_freq()
        return self.original_base

    # source: https://github.com/huggingface/transformers/blob/d5a99dfcee6e94065cb7c83cc8ab6fc5daa0cc4e/src/transformers/modeling_rope_utils.py#L318
    def compute_llama3_base_freq(self):
        factor = self.rope_scaling["factor"]
        low_freq_factor = self.rope_scaling.get("low_freq_factor", 1.0)
        high_freq_factor = self.rope_scaling.get("high_freq_factor", 4.0)
        old_context_len = self.rope_scaling.get(
            "original_max_position_embeddings",
            8192,
        )

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        freqs = self.original_base ** (mx.arange(0, self.dims, 2) / self.dims)
        wavelens = 2 * mx.pi * freqs
        new_base_freqs = []

        smooths = (wavelens - high_freq_wavelen) / (
            low_freq_wavelen - high_freq_wavelen
        )
        new_base_freqs = freqs * (1 - smooths) * factor + smooths
        new_base_freqs = mx.where(wavelens < high_freq_wavelen, freqs, new_base_freqs)
        new_base_freqs = mx.where(
            wavelens > low_freq_wavelen, freqs * factor, new_base_freqs
        )
        return new_base_freqs.mean().item()

    def extra_repr(self):
        return (
            f"{self.dims}, traditional={self.traditional}, "
            f"max_position_embeddings={self.max_position_embeddings}, "
            f"scaling_factor={self.scale}, rope_type={self.rope_type}"
        )

    def __call__(self, x, offset: int = 0):
        seq_len = x.shape[1] + offset
        base = self.base
        if self.max_position_embeddings and seq_len > self.max_position_embeddings:
            base *= (
                (self.scale * seq_len / self.max_position_embeddings) - (self.scale - 1)
            ) ** (self.dims / (self.dims - 2))

        return mx.fast.rope(
            x,
            self.dims,
            traditional=self.traditional,
            base=base,
            scale=self.scale,
            offset=offset,
        )


def initialize_rope(args: ModelArgs):
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
            rope_scale = 1.0  # The scaling is handled internally for llama3

    return DynamicNTKScalingRoPE(
        dims=head_dim,
        max_position_embeddings=args.max_position_embeddings,
        traditional=args.rope_traditional,
        base=args.rope_theta,
        scale=rope_scale,
        rope_type=rope_type,
        rope_scaling=rope_scaling,
    )

