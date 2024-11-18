from typing import List, Dict, Optional, Union, Tuple
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from dataclasses import dataclass

from core.model import Model
from utils.loading import TokenizerWrapper
from utils.logging import setup_logger

logger = setup_logger(__name__)

@dataclass
class TokenAnalysisResult:
    """Container for token analysis results."""
    token_ids: List[int]
    token_text: List[str]
    embeddings: mx.array
    layer_states: Optional[List[mx.array]] = None
    attention_patterns: Optional[List[mx.array]] = None
    neuron_activations: Optional[Dict[int, mx.array]] = None

class TokenAnalyzer:
    """Analyzes token representations and activations."""
    
    def __init__(self, model: Model, tokenizer: TokenizerWrapper):
        self.model = model
        self.tokenizer = tokenizer
        
    def analyze_sequence(
        self,
        text: str,
        return_layer_states: bool = True,
        return_attention: bool = True,
        return_neuron_activations: bool = True
    ) -> TokenAnalysisResult:
        """Analyze token representations for a text sequence."""
        # Tokenize input
        tokens = self.tokenizer(text, return_tensors="np")
        token_ids = tokens["input_ids"][0]
        
        # Get token texts
        token_text = [self.tokenizer.decode([tid]) for tid in token_ids]
        
        # Forward pass with all states
        outputs = self.model(
            mx.array(tokens["input_ids"]),
            return_hidden_states=return_layer_states,
            return_attention=return_attention
        )
        
        # Extract embeddings and states
        if isinstance(outputs, tuple):
            embeddings = outputs[0]  # Use first hidden state as embeddings
            hidden_states = outputs[1] if return_layer_states else None
            attention_patterns = outputs[2] if return_attention else None
        else:
            embeddings = outputs
            hidden_states = None
            attention_patterns = None
            
        # Get neuron activations if requested
        neuron_activations = None
        if return_neuron_activations and hidden_states is not None:
            neuron_activations = self._extract_neuron_activations(hidden_states)
            
        return TokenAnalysisResult(
            token_ids=token_ids.tolist(),
            token_text=token_text,
            embeddings=embeddings,
            layer_states=hidden_states,
            attention_patterns=attention_patterns,
            neuron_activations=neuron_activations
        )
    
    def compare_sequences(
        self,
        text1: str,
        text2: str,
        method: str = "cosine",
        layer: int = -1
    ) -> mx.array:
        """Compare token representations between two sequences."""
        # Get embeddings
        analysis1 = self.analyze_sequence(text1)
        analysis2 = self.analyze_sequence(text2)
        
        # Get representations from specified layer
        if layer == -1:
            rep1 = analysis1.embeddings
            rep2 = analysis2.embeddings
        else:
            rep1 = analysis1.layer_states[layer]
            rep2 = analysis2.layer_states[layer]
        
        # Compute similarity matrix
        if method == "cosine":
            sim = self._cosine_similarity(rep1, rep2)
        elif method == "euclidean":
            sim = self._euclidean_distance(rep1, rep2)
        else:
            raise ValueError(f"Unsupported similarity method: {method}")
            
        return sim
    
    def _extract_neuron_activations(
        self,
        hidden_states: List[mx.array]
    ) -> Dict[int, mx.array]:
        """Extract neuron activations from hidden states."""
        activations = {}
        for layer_idx, layer_state in enumerate(hidden_states):
            # Get MLP activations
            mlp = self.model.layers[layer_idx].mlp
            gate = mlp.gate_proj(layer_state)
            up = mlp.up_proj(layer_state)
            
            # Apply SwiGLU activation
            activated = nn.silu(gate) * up
            
            # Store activations
            activations[layer_idx] = activated
            
        return activations
    
    def _cosine_similarity(self, a: mx.array, b: mx.array) -> mx.array:
        """Compute cosine similarity between two sets of vectors."""
        # Reshape inputs to 2D: (batch_size * seq_len, hidden_size)
        a_flat = a.reshape(-1, a.shape[-1])
        b_flat = b.reshape(-1, b.shape[-1])
        
        # Compute norms
        a_norm = mx.sqrt(mx.sum(a_flat * a_flat, axis=-1, keepdims=True))
        b_norm = mx.sqrt(mx.sum(b_flat * b_flat, axis=-1, keepdims=True))
        
        # Compute similarity
        sim = mx.matmul(a_flat, b_flat.transpose())
        norm_product = mx.matmul(a_norm, b_norm.transpose())
        
        # Reshape back to (seq_len1, seq_len2)
        return sim / (norm_product + 1e-8)
    
    def _euclidean_distance(self, a: mx.array, b: mx.array) -> mx.array:
        """Compute Euclidean distance between two sets of vectors."""
        # Reshape inputs to 2D: (batch_size * seq_len, hidden_size)
        a_flat = a.reshape(-1, a.shape[-1])
        b_flat = b.reshape(-1, b.shape[-1])
        
        # Compute pairwise distances
        a_norm = mx.sum(a_flat * a_flat, axis=1)
        b_norm = mx.sum(b_flat * b_flat, axis=1)
        
        dist = a_norm[:, None] + b_norm[None, :] - 2.0 * mx.matmul(a_flat, b_flat.transpose())
        return mx.sqrt(mx.maximum(dist, 0.0))