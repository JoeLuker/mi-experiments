import mlx.core as mx
import mlx.nn as nn
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.intermediate_size = args.intermediate_size
        
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=args.mlp_bias)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=args.mlp_bias)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=args.mlp_bias)
        
        # Initialize neuron scaling factors (for emphasis/ablation)
        self.neuron_scale = mx.array([1.0] * args.intermediate_size, dtype=mx.float32)

    def forward(self, x: mx.array) -> mx.array:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        # Apply neuron scaling (including ablation if scaling is zero)
        neuron_scale = self.neuron_scale[None, None, :]
        gate = gate * neuron_scale
        up = up * neuron_scale
        
        # SwiGLU activation
        intermediate = mx.sigmoid(gate) * up
        
        return self.down_proj(intermediate)
