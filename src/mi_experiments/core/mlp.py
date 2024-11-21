import mlx.core as mx
import mlx.nn as nn

from mi_experiments.core.config import ModelArgs


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        hidden_dim = args.intermediate_size
        if hasattr(args, "mlp_bias"):
            mlp_bias = args.mlp_bias
        else:
            mlp_bias = False

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=mlp_bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=mlp_bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=mlp_bias)
        
        # Emphasis scaling factors for neurons
        self.neuron_scale = mx.ones((hidden_dim,))  # Initialize neuron scaling

    def __call__(self, x) -> mx.array:
        gate_output = nn.silu(self.gate_proj(x))
        up_output = self.up_proj(x)
        
        # Apply neuron scaling
        scale = self.neuron_scale.reshape(1, 1, -1)  # [1, 1, hidden_dim]
        scaled_output = (gate_output * up_output) * scale
        
        return self.down_proj(scaled_output)
