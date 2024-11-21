import mlx.core as mx
import mlx.nn as nn

from mi_experiments.core.config import ModelArgs
from mi_experiments.core.blocks import TransformerBlock
from mi_experiments.core.attention import create_attention_mask
from typing import Dict, Any, List


class LlamaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        return_hidden_states: bool = False,
    ):
        h = self.embed_tokens(inputs)

        mask = create_attention_mask(h, cache)

        if cache is None:
            cache = [None] * len(self.layers)

        hidden_states = []
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)
            if return_hidden_states:
                hidden_states.append(h)

        h = self.norm(h)

        if return_hidden_states:
            return h, hidden_states
        return h

class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LlamaModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        return_hidden_states: bool = False,
    ):
        model_output = self.model(inputs, cache, return_hidden_states)
        
        if return_hidden_states:
            h, hidden_states = model_output
        else:
            h = model_output

        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(h)
        else:
            out = self.lm_head(h)

        if return_hidden_states:
            return out, hidden_states
        return out

    def sanitize(self, weights):
        # Remove unused precomputed rotary freqs and handle missing emphasis params
        sanitized = {
            k: v for k, v in weights.items() 
            if "self_attn.rotary_emb.inv_freq" not in k
        }
        
        # Initialize missing emphasis parameters
        for i in range(len(self.model.layers)):
            layer = f"model.layers.{i}"
            if f"{layer}.layer_scale" not in sanitized:
                sanitized[f"{layer}.layer_scale"] = mx.ones(1, dtype=mx.float32)
            if f"{layer}.mlp.neuron_scale" not in sanitized:
                sanitized[f"{layer}.mlp.neuron_scale"] = mx.ones(
                    self.model.layers[i].mlp.gate_proj.weight.shape[0], 
                    dtype=mx.float32
                )
        
        return sanitized

    def set_emphasis_config(self, emphasis_config: Dict[str, Any]):
        """
        Sets the emphasis configuration for layers, heads, and neurons.
        Setting emphasis to zero effectively ablates the component.
        
        Args:
            emphasis_config: Dictionary with format:
                {
                    'layers': {'1': 0.0, '2': 0.0, '0': 1.5},
                    'heads': {'3': {'0': 0.0, '2': 0.0, '1': 2.0}},
                    'neurons': {'4': {'10': 0.0, '20': 0.0, '30': 0.0, '15': 1.5}}
                }
        """
        # Set layer scaling factors
        layers_config = emphasis_config.get('layers', {})
        for idx_str, scaling_factor in layers_config.items():
            idx = int(idx_str)
            if 0 <= idx < len(self.model.layers):
                scaling_factor_array = mx.array(scaling_factor, dtype=mx.float32)
                self.model.layers[idx].layer_scale = scaling_factor_array

        # Set head scaling factors
        heads_config = emphasis_config.get('heads', {})
        for layer_idx_str, heads_info in heads_config.items():
            layer_idx = int(layer_idx_str)
            if 0 <= layer_idx < len(self.model.layers):
                layer = self.model.layers[layer_idx]
                # Create new array for head scales
                new_head_scale = mx.ones((layer.self_attn.n_heads,), dtype=mx.float32)
                for head_idx_str, scaling_factor in heads_info.items():
                    head_idx = int(head_idx_str)
                    if 0 <= head_idx < layer.self_attn.n_heads:
                        # Update individual head scale using array indexing
                        new_head_scale[head_idx] = scaling_factor
                layer.self_attn.head_scale = new_head_scale

        # Set neuron scaling factors
        neurons_config = emphasis_config.get('neurons', {})
        for layer_idx_str, neurons_info in neurons_config.items():
            layer_idx = int(layer_idx_str)
            if 0 <= layer_idx < len(self.model.layers):
                layer = self.model.layers[layer_idx]
                # Create new array for neuron scales
                new_neuron_scale = mx.ones((layer.mlp.gate_proj.weight.shape[0],), dtype=mx.float32)
                for neuron_idx_str, scaling_factor in neurons_info.items():
                    neuron_idx = int(neuron_idx_str)
                    if 0 <= neuron_idx < layer.mlp.gate_proj.weight.shape[0]:
                        # Update individual neuron scale using array indexing
                        new_neuron_scale[neuron_idx] = scaling_factor
                layer.mlp.neuron_scale = new_neuron_scale

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return (
            self.args.head_dim or self.args.hidden_size // self.args.num_attention_heads
        )

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads

    def get_layer_output(self, layer_idx: int, inputs: mx.array) -> mx.array:
        """Get the output of a specific layer.
        
        Args:
            layer_idx: Index of the layer to get output from
            inputs: Input tokens
            
        Returns:
            Layer output tensor
        """
        if not 0 <= layer_idx < len(self.model.layers):
            raise ValueError(f"Layer index {layer_idx} out of range [0, {len(self.model.layers)})")
        
        # Get embeddings
        h = self.model.embed_tokens(inputs)
        
        # Process through layers up to requested layer
        for i, layer in enumerate(self.model.layers):
            h = layer(h)
            if i == layer_idx:
                return self.model.norm(h)  # Apply final norm
            
        return h
    
    def get_all_layer_outputs(self, inputs: mx.array) -> List[mx.array]:
        """Get the output of all layers."""
        return [self.get_layer_output(i, inputs) for i in range(len(self.model.layers))]