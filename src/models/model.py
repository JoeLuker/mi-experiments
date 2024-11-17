from dataclasses import dataclass
from typing import Optional, Dict, Union, List
import mlx.core as mx
import mlx.nn as nn
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

@dataclass
class ModelConfig:
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    head_dim: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    attention_bias: bool = False
    mlp_bias: bool = False
    rope_theta: float = 10000
    rope_traditional: bool = False
    tie_word_embeddings: bool = True

    @classmethod
    def from_model(cls, model):
        return cls(
            model_type=model.model.args.model_type,
            hidden_size=model.model.args.hidden_size,
            num_hidden_layers=model.model.args.num_hidden_layers,
            intermediate_size=model.model.args.intermediate_size,
            num_attention_heads=model.model.args.num_attention_heads,
            rms_norm_eps=model.model.args.rms_norm_eps,
            vocab_size=model.model.args.vocab_size,
            head_dim=model.head_dim,
            num_key_value_heads=model.n_kv_heads,
            tie_word_embeddings=model.model.args.tie_word_embeddings
        )

class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def set_emphasis_config(self, emphasis_config: Dict[str, Any]):
        """Apply emphasis configuration to model components"""
        from ..inference.emphasis import (
            apply_layer_emphasis,
            apply_head_emphasis,
            apply_neuron_emphasis
        )
        
        if 'layers' in emphasis_config:
            apply_layer_emphasis(self, emphasis_config['layers'])
        if 'heads' in emphasis_config:
            apply_head_emphasis(self, emphasis_config['heads'])
        if 'neurons' in emphasis_config:
            apply_neuron_emphasis(self, emphasis_config['neurons'])

    def forward(self, inputs: mx.array, cache=None, return_hidden_states: bool = False):
        h = self.embed_tokens(inputs)
        
        if cache is None:
            cache = [None] * len(self.layers)
            
        hidden_states = []
        for layer, c in zip(self.layers, cache):
            h, attn_weights, _, _, new_cache = layer(h, cache=c)
            if return_hidden_states:
                hidden_states.append(h)
                
        h = self.norm(h)
        
        if self.config.tie_word_embeddings:
            out = self.embed_tokens.as_linear(h)
        else:
            out = self.lm_head(h)
            
        if return_hidden_states:
            return out, hidden_states, attn_weights
        return out 