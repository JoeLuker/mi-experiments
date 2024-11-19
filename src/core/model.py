# core/model.py

from typing import Any, Dict, List, Optional, Union, Tuple

import mlx.core as mx
import mlx.nn as nn

from .blocks import TransformerBlock
from .config import ModelArgs
from .cache import BatchedKVCache


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.vocab_size = args.vocab_size
        
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        else:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
            self.lm_head.weight = self.embed_tokens.weight

    def __call__(
        self,
        inputs: mx.array,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[Union[List[Any], BatchedKVCache]] = None,
        return_hidden_states: bool = False,
    ):
        h = self.embed_tokens(inputs)
        hidden_states = []

        # Handle cache initialization
        if isinstance(cache, BatchedKVCache):
            cache = [cache] * len(self.layers)
        elif cache is None:
            cache = [None] * len(self.layers)

        # Process layers
        for i, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
            h = layer(h, mask=attention_mask, cache=layer_cache)
            if return_hidden_states:
                hidden_states.append(h)

        h = self.norm(h)
        logits = self.lm_head(h)

        if return_hidden_states:
            return logits, hidden_states
        return logits
