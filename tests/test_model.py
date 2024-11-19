import pytest
import mlx.core as mx
import numpy as np
from typing import Dict, List, Optional

from src.core.model import Model, BatchedKVCache
from src.core.config import ModelArgs
from src.core.blocks import TransformerBlock, MLP
from src.utils.loading import load_config, get_model_path

class MockTokenizer:
    def __init__(self):
        self.vocab_size = 100
        self.pad_token_id = 0
        
    def __call__(self, text: str, return_tensors: str = "np") -> Dict:
        # Mock tokenization
        ids = [ord(c) % self.vocab_size for c in text]
        if return_tensors == "np":
            return {"input_ids": np.array([ids])}
        return {"input_ids": ids}
    
    def decode(self, ids: List[int]) -> str:
        return chr(ids[0] % self.vocab_size)

@pytest.fixture
def model_args():
    return ModelArgs(
        model_type="llama",
        hidden_size=32,
        intermediate_size=128,
        num_attention_heads=4,
        num_hidden_layers=2,
        vocab_size=100,
        rms_norm_eps=1e-5,
        max_position_embeddings=128,
        attention_bias=True,
        num_key_value_heads=None
    )

@pytest.fixture
def model(model_args):
    return Model(model_args)

def test_model_initialization(model_args):
    model = Model(model_args)
    
    # Check basic attributes
    assert len(model.layers) == model_args.num_hidden_layers
    assert model.embed_tokens.weight.shape == (model_args.vocab_size, model_args.hidden_size)
    
    # Check layer components
    for layer in model.layers:
        assert isinstance(layer, TransformerBlock)
        assert isinstance(layer.mlp, MLP)
        assert layer.self_attn.n_heads == model_args.num_attention_heads
        assert layer.self_attn.head_dim == model_args.hidden_size // model_args.num_attention_heads

def test_forward_pass(model):
    batch_size = 2
    seq_length = 10
    input_ids = mx.random.randint(0, 100, (batch_size, seq_length))
    
    # Basic forward pass
    outputs = model(input_ids)
    assert outputs.shape == (batch_size, seq_length, model.args.vocab_size)
    
    # Test with attention mask
    attention_mask = mx.ones((batch_size, seq_length))
    outputs_masked = model(input_ids, attention_mask=attention_mask)
    assert outputs_masked.shape == (batch_size, seq_length, model.args.vocab_size)
    
    # Test with hidden states return
    outputs, hidden_states = model(input_ids, return_hidden_states=True)
    assert len(hidden_states) == len(model.layers)
    for states in hidden_states:
        assert states.shape == (batch_size, seq_length, model.args.hidden_size)

def test_kv_cache(model):
    batch_size = 1
    seq_length = 5
    input_ids = mx.random.randint(0, 100, (batch_size, seq_length))
    
    # Initialize cache
    cache = BatchedKVCache(
        head_dim=model.args.hidden_size // model.args.num_attention_heads,
        n_kv_heads=model.args.num_attention_heads,
        batch_size=batch_size
    )
    
    # Forward pass with cache
    outputs_with_cache = model(input_ids, cache=cache)
    assert outputs_with_cache.shape == (batch_size, seq_length, model.args.vocab_size)
    assert cache.offset == seq_length
    
    # Test cache update
    next_token = mx.random.randint(0, 100, (batch_size, 1))
    outputs_next = model(next_token, cache=cache)
    assert outputs_next.shape == (batch_size, 1, model.args.vocab_size)
    assert cache.offset == seq_length + 1

def test_model_generation(model):
    tokenizer = MockTokenizer()
    prompt = "Hello"
    input_ids = mx.array(tokenizer(prompt, return_tensors="np")["input_ids"])
    
    # Test basic generation
    max_new_tokens = 5
    generated = []
    cache = BatchedKVCache(
        head_dim=model.args.hidden_size // model.args.num_attention_heads,
        n_kv_heads=model.args.num_attention_heads
    )
    
    # Initial forward pass
    outputs = model(input_ids, cache=cache)
    next_token = mx.argmax(outputs[:, -1:, :], axis=-1)
    generated.append(next_token)
    
    # Generate remaining tokens
    for _ in range(max_new_tokens - 1):
        outputs = model(next_token, cache=cache)
        next_token = mx.argmax(outputs[:, -1:, :], axis=-1)
        generated.append(next_token)
    
    generated_tokens = mx.concatenate(generated, axis=1)
    assert generated_tokens.shape == (1, max_new_tokens)

def test_model_config_validation():
    # Test invalid hidden size
    with pytest.raises(ValueError):
        ModelArgs(
            model_type="llama",
            hidden_size=-32,
            intermediate_size=128,
            num_attention_heads=4,
            num_hidden_layers=2,
            vocab_size=100,
            rms_norm_eps=1e-5
        )
    
    # Test invalid number of layers
    with pytest.raises(ValueError):
        ModelArgs(
            model_type="llama",
            hidden_size=32,
            intermediate_size=128,
            num_attention_heads=4,
            num_hidden_layers=0,
            vocab_size=100,
            rms_norm_eps=1e-5
        )
    
    # Test invalid head configuration
    with pytest.raises(ValueError):
        ModelArgs(
            model_type="llama",
            hidden_size=32,
            intermediate_size=128,
            num_attention_heads=3,
            num_hidden_layers=2,
            vocab_size=100,
            rms_norm_eps=1e-5
        )

def test_attention_bias(model_args):
    # Test with and without attention bias
    model_with_bias = Model(model_args)
    model_args.attention_bias = False
    model_without_bias = Model(model_args)
    
    # Check bias parameters
    for layer in model_with_bias.layers:
        assert layer.self_attn.q_proj.get('bias') is not None
        assert layer.self_attn.k_proj.get('bias') is not None
        assert layer.self_attn.v_proj.get('bias') is not None
        
    for layer in model_without_bias.layers:
        assert layer.self_attn.q_proj.get('bias') is None
        assert layer.self_attn.k_proj.get('bias') is None
        assert layer.self_attn.v_proj.get('bias') is None
