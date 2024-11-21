import mlx.core as mx
from typing import Dict, Any, Union, Optional

class EmphasisConfig:
    """Configuration class for model component emphasis"""
    
    def __init__(
        self,
        layers: Optional[Dict[Union[str, int], float]] = None,
        heads: Optional[Dict[Union[str, int], Dict[Union[str, int], float]]] = None,
        neurons: Optional[Dict[Union[str, int], Dict[Union[str, int], float]]] = None
    ):
        self.layers = layers or {}
        self.heads = heads or {}
        self.neurons = neurons or {}
        
    def validate(self, model):
        """Validate configuration against model architecture"""
        num_layers = len(model.layers)
        
        # Validate layer indices
        for layer_idx in self.layers:
            if int(layer_idx) >= num_layers:
                raise ValueError(f"Layer index {layer_idx} exceeds model layers ({num_layers})")
        
        # Validate head indices
        for layer_idx, heads_info in self.heads.items():
            if int(layer_idx) >= num_layers:
                raise ValueError(f"Layer index {layer_idx} exceeds model layers ({num_layers})")
            num_heads = model.layers[int(layer_idx)].self_attn.n_heads
            for head_idx in heads_info:
                if int(head_idx) >= num_heads:
                    raise ValueError(f"Head index {head_idx} exceeds model heads ({num_heads})")
        
        # Validate neuron indices
        for layer_idx, neurons_info in self.neurons.items():
            if int(layer_idx) >= num_layers:
                raise ValueError(f"Layer index {layer_idx} exceeds model layers ({num_layers})")
            hidden_dim = model.layers[int(layer_idx)].mlp.gate_proj.weight.shape[0]
            for neuron_idx in neurons_info:
                if int(neuron_idx) >= hidden_dim:
                    raise ValueError(f"Neuron index {neuron_idx} exceeds hidden dimension ({hidden_dim})")

def apply_emphasis_config(model, config: Dict[str, Any]):
    """Apply emphasis configuration to model components"""
    emphasis_config = EmphasisConfig(**config)
    emphasis_config.validate(model)
    
    # Apply layer scaling
    for layer_idx_str, scale in emphasis_config.layers.items():
        layer_idx = int(layer_idx_str)
        model.layers[layer_idx].layer_scale = mx.array(scale, dtype=mx.float32)
    
    # Apply head scaling
    for layer_idx_str, heads_info in emphasis_config.heads.items():
        layer_idx = int(layer_idx_str)
        for head_idx_str, scale in heads_info.items():
            head_idx = int(head_idx_str)
            model.layers[layer_idx].self_attn.head_scale = model.layers[layer_idx].self_attn.head_scale.at[head_idx].set(scale)
    
    # Apply neuron scaling
    for layer_idx_str, neurons_info in emphasis_config.neurons.items():
        layer_idx = int(layer_idx_str)
        for neuron_idx_str, scale in neurons_info.items():
            neuron_idx = int(neuron_idx_str)
            model.layers[layer_idx].mlp.neuron_scale = model.layers[layer_idx].mlp.neuron_scale.at[neuron_idx].set(scale)
