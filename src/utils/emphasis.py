# utils/emphasis.py

from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
import json
from pathlib import Path

import mlx.core as mx

@dataclass
class EmphasisConfig:
    """Configuration for model emphasis/ablation."""
    
    layers: Dict[int, float] = None  # layer_idx -> scaling_factor
    heads: Dict[int, Dict[int, float]] = None  # layer_idx -> {head_idx -> scaling_factor}
    neurons: Dict[int, Dict[int, float]] = None  # layer_idx -> {neuron_idx -> scaling_factor}
    
    def __post_init__(self):
        """Initialize empty dictionaries and validate configuration."""
        self.layers = self.layers or {}
        self.heads = self.heads or {}
        self.neurons = self.neurons or {}
        self.validate()
    
    def validate(self) -> None:
        """Validate emphasis configuration."""
        # Validate layer scaling
        for layer_idx, scale in self.layers.items():
            if not isinstance(layer_idx, int) or layer_idx < 0:
                raise ValueError(f"Invalid layer index: {layer_idx}")
            if not isinstance(scale, (int, float)):
                raise ValueError(f"Invalid scaling factor for layer {layer_idx}: {scale}")
                
        # Validate head scaling
        for layer_idx, heads in self.heads.items():
            if not isinstance(layer_idx, int) or layer_idx < 0:
                raise ValueError(f"Invalid layer index: {layer_idx}")
            for head_idx, scale in heads.items():
                if not isinstance(head_idx, int) or head_idx < 0:
                    raise ValueError(f"Invalid head index: {head_idx}")
                if not isinstance(scale, (int, float)):
                    raise ValueError(f"Invalid scaling factor for head {head_idx}: {scale}")
                    
        # Validate neuron scaling
        for layer_idx, neurons in self.neurons.items():
            if not isinstance(layer_idx, int) or layer_idx < 0:
                raise ValueError(f"Invalid layer index: {layer_idx}")
            for neuron_idx, scale in neurons.items():
                if not isinstance(neuron_idx, int) or neuron_idx < 0:
                    raise ValueError(f"Invalid neuron index: {neuron_idx}")
                if not isinstance(scale, (int, float)):
                    raise ValueError(f"Invalid scaling factor for neuron {neuron_idx}: {scale}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "layers": self.layers,
            "heads": self.heads,
            "neurons": self.neurons
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EmphasisConfig":
        """Create configuration from dictionary."""
        return cls(
            layers=config_dict.get("layers", {}),
            heads=config_dict.get("heads", {}),
            neurons=config_dict.get("neurons", {})
        )
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "EmphasisConfig":
        """Load configuration from file."""
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

class EmphasisManager:
    """Manager for applying emphasis/ablation configurations."""
    
    def __init__(self, model):
        self.model = model
        self.current_config = EmphasisConfig()
        
    def apply_config(self, config: EmphasisConfig) -> None:
        """Apply emphasis configuration to model."""
        # Validate configuration against model
        self._validate_model_compatibility(config)
        
        # Apply layer scaling
        for layer_idx, scale in config.layers.items():
            self.model.layers[layer_idx].layer_scale = mx.array(scale, dtype=mx.float32)
            
        # Apply head scaling
        for layer_idx, heads in config.heads.items():
            layer = self.model.layers[layer_idx]
            for head_idx, scale in heads.items():
                layer.self_attn.head_scale[head_idx] = mx.array(scale, dtype=mx.float32)
                
        # Apply neuron scaling
        for layer_idx, neurons in config.neurons.items():
            layer = self.model.layers[layer_idx]
            for neuron_idx, scale in neurons.items():
                layer.mlp.neuron_scale[neuron_idx] = mx.array(scale, dtype=mx.float32)
        
        self.current_config = config
    
    def _validate_model_compatibility(self, config: EmphasisConfig) -> None:
        """Validate configuration compatibility with model."""
        num_layers = len(self.model.layers)
        num_heads = self.model.layers[0].self_attn.n_heads
        hidden_dim = self.model.layers[0].mlp.neuron_scale.shape[0]
        
        # Check layer indices
        for layer_idx in config.layers.keys():
            if layer_idx >= num_layers:
                raise ValueError(f"Layer index {layer_idx} exceeds model layers ({num_layers})")
                
        # Check head indices
        for layer_idx, heads in config.heads.items():
            if layer_idx >= num_layers:
                raise ValueError(f"Layer index {layer_idx} exceeds model layers ({num_layers})")
            for head_idx in heads.keys():
                if head_idx >= num_heads:
                    raise ValueError(f"Head index {head_idx} exceeds model heads ({num_heads})")
                    
        # Check neuron indices
        for layer_idx, neurons in config.neurons.items():
            if layer_idx >= num_layers:
                raise ValueError(f"Layer index {layer_idx} exceeds model layers ({num_layers})")
            for neuron_idx in neurons.keys():
                if neuron_idx >= hidden_dim:
                    raise ValueError(f"Neuron index {neuron_idx} exceeds hidden dimension ({hidden_dim})")
    
    def reset(self) -> None:
        """Reset all emphasis/ablation to default values."""
        default_config = EmphasisConfig()
        self.apply_config(default_config)