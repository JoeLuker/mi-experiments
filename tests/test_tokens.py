import pytest
import mlx.core as mx
import numpy as np
from typing import List, Dict

from src.analysis.tokens import TokenAnalyzer, TokenAnalysisResult
from src.core.model import Model, ModelArgs
from src.utils.loading import TokenizerWrapper

class MockTokenizer:
    def __init__(self):
        self.vocab_size = 100
        
    def __call__(self, text: str, return_tensors: str = "np") -> Dict:
        # Mock tokenization - just convert chars to numbers
        ids = [ord(c) % self.vocab_size for c in text]
        if return_tensors == "np":
            return {"input_ids": np.array([ids])}
        return {"input_ids": ids}
    
    def decode(self, ids: List[int]) -> str:
        return chr(ids[0] % self.vocab_size)

class MockModel:
    def __init__(self):
        self.hidden_size = 32
        self.num_layers = 2
        self.vocab_size = 100
        
        # Mock layers with MLPs
        self.layers = [MockLayer(self.hidden_size) for _ in range(self.num_layers)]
        
    def __call__(
        self,
        inputs: mx.array,
        return_hidden_states: bool = False,
        return_attention: bool = False
    ):
        batch_size, seq_len = inputs.shape
        
        # Generate mock embeddings with hidden_size dimension
        h = mx.random.normal((batch_size, seq_len, self.hidden_size))
        hidden_states = []
        attention_patterns = []
        
        # Pass through layers
        for layer in self.layers:
            h = layer(h)
            if return_hidden_states:
                hidden_states.append(h)
            if return_attention:
                attention_patterns.append(
                    mx.random.normal((batch_size, 8, seq_len, seq_len))
                )
        
        # Final projection to vocab_size dimension
        logits = mx.random.normal((batch_size, seq_len, self.vocab_size))
        
        if return_hidden_states and return_attention:
            return h, hidden_states, attention_patterns  # Return h instead of logits
        elif return_hidden_states:
            return h, hidden_states  # Return h instead of logits
        elif return_attention:
            return h, None, attention_patterns  # Return h instead of logits
        return h  # Return h instead of logits

class MockLayer:
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
        self.mlp = MockMLP(hidden_size)
        
    def __call__(self, x: mx.array) -> mx.array:
        return self.mlp(x)

class MockMLP:
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
        self.intermediate_size = hidden_size * 4
        
        # Mock projections
        self.gate_proj = lambda x: mx.random.normal((*x.shape[:-1], self.intermediate_size))
        self.up_proj = lambda x: mx.random.normal((*x.shape[:-1], self.intermediate_size))
        
    def __call__(self, x: mx.array) -> mx.array:
        return mx.random.normal(x.shape)

@pytest.fixture
def analyzer():
    model = MockModel()
    tokenizer = MockTokenizer()
    return TokenAnalyzer(model, tokenizer)

def test_analyze_sequence_basic(analyzer):
    text = "Hello world"
    result = analyzer.analyze_sequence(
        text,
        return_layer_states=False,
        return_attention=False,
        return_neuron_activations=False
    )
    
    assert isinstance(result, TokenAnalysisResult)
    assert len(result.token_ids) == len(text)
    assert len(result.token_text) == len(text)
    assert result.embeddings.shape[1] == len(text)  # Check sequence length
    assert result.embeddings.shape[2] == analyzer.model.hidden_size  # Check hidden size
    assert result.layer_states is None
    assert result.attention_patterns is None
    assert result.neuron_activations is None

def test_analyze_sequence_full(analyzer):
    text = "Hello world"
    result = analyzer.analyze_sequence(
        text,
        return_layer_states=True,
        return_attention=True,
        return_neuron_activations=True
    )
    
    assert isinstance(result, TokenAnalysisResult)
    assert len(result.layer_states) == analyzer.model.num_layers
    assert len(result.attention_patterns) == analyzer.model.num_layers
    assert len(result.neuron_activations) == analyzer.model.num_layers
    
    # Check shapes
    seq_len = len(text)
    hidden_size = analyzer.model.hidden_size
    
    for layer_state in result.layer_states:
        assert layer_state.shape == (1, seq_len, hidden_size)
        
    for attn in result.attention_patterns:
        assert attn.shape == (1, 8, seq_len, seq_len)
        
    for layer_idx, activations in result.neuron_activations.items():
        assert activations.shape == (1, seq_len, hidden_size * 4)

def test_compare_sequences(analyzer):
    text1 = "Hello"
    text2 = "World"
    
    # Test cosine similarity
    sim_cos = analyzer.compare_sequences(text1, text2, method="cosine")
    assert sim_cos.shape == (len(text1), len(text2))
    assert mx.all((sim_cos >= -1.0) & (sim_cos <= 1.0))
    
    # Test euclidean distance
    sim_euc = analyzer.compare_sequences(text1, text2, method="euclidean")
    assert sim_euc.shape == (len(text1), len(text2))
    assert mx.all(sim_euc >= 0.0)
    
    # Test invalid method
    with pytest.raises(ValueError):
        analyzer.compare_sequences(text1, text2, method="invalid")

def test_compare_sequences_layer_specific(analyzer):
    text1 = "Hello"
    text2 = "World"
    
    # Test specific layer comparison
    for layer in range(analyzer.model.num_layers):
        sim = analyzer.compare_sequences(text1, text2, layer=layer)
        assert sim.shape == (len(text1), len(text2))

def test_neuron_activations(analyzer):
    text = "Test"
    result = analyzer.analyze_sequence(
        text,
        return_neuron_activations=True
    )
    
    assert result.neuron_activations is not None
    assert len(result.neuron_activations) == analyzer.model.num_layers
    
    # Check activation shapes and values
    for layer_idx, activations in result.neuron_activations.items():
        # Shape should be (batch_size, seq_len, intermediate_size)
        assert activations.shape == (1, len(text), analyzer.model.hidden_size * 4)
        
        # Activations should be finite
        assert mx.all(mx.isfinite(activations))