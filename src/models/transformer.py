from dataclasses import dataclass
from typing import Optional, Dict, Union, List
import mlx.core as mx
import mlx.nn as nn
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

@dataclass
class ModelArgs:
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
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

class TransformerModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None, return_hidden_states: bool = False):
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
        
        if self.args.tie_word_embeddings:
            out = self.embed_tokens.as_linear(h)
        else:
            out = self.lm_head(h)
            
        if return_hidden_states:
            return out, hidden_states
        return out
