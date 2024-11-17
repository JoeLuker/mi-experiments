from typing import Dict, Any
import mlx.core as mx

def apply_layer_emphasis(model, layer_config: Dict[str, float]):
    """Apply emphasis/ablation to model layers."""
    for layer_idx_str, scaling_factor in layer_config.items():
        layer_idx = int(layer_idx_str)
        if 0 <= layer_idx < len(model.layers):
            scaling_factor_array = mx.array(scaling_factor, dtype=mx.float32)
            model.layers[layer_idx].layer_scale = scaling_factor_array

def apply_head_emphasis(model, head_config: Dict[str, Dict[str, float]]):
    """Apply emphasis/ablation to attention heads."""
    for layer_idx_str, heads_info in head_config.items():
        layer_idx = int(layer_idx_str)
        if 0 <= layer_idx < len(model.layers):
            head_scale = model.layers[layer_idx].self_attn.head_scale
            for head_idx_str, scaling_factor in heads_info.items():
                head_idx = int(head_idx_str)
                if 0 <= head_idx < len(head_scale):
                    scaling_factor_array = mx.array(
                        scaling_factor, 
                        dtype=head_scale.dtype
                    )
                    head_scale[head_idx] = scaling_factor_array
            model.layers[layer_idx].self_attn.head_scale = head_scale

def apply_neuron_emphasis(model, neuron_config: Dict[str, Dict[str, float]]):
    """Apply emphasis/ablation to MLP neurons."""
    for layer_idx_str, neurons_info in neuron_config.items():
        layer_idx = int(layer_idx_str)
        if 0 <= layer_idx < len(model.layers):
            neuron_scale = model.layers[layer_idx].mlp.neuron_scale
            for neuron_idx_str, scaling_factor in neurons_info.items():
                neuron_idx = int(neuron_idx_str)
                if 0 <= neuron_idx < len(neuron_scale):
                    scaling_factor_array = mx.array(
                        scaling_factor, 
                        dtype=neuron_scale.dtype
                    )
                    neuron_scale[neuron_idx] = scaling_factor_array
            model.layers[layer_idx].mlp.neuron_scale = neuron_scale