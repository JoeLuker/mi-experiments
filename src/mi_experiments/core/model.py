import mlx.core as mx
import mlx.nn as nn

from mi_experiments.core.config import ModelArgs
from mi_experiments.core.blocks import TransformerBlock
from mi_experiments.core.attention import create_attention_mask


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
        # Remove unused precomputed rotary freqs
        return {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }

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
