# analysis/attention.py

import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import mlx.core as mx

from ..core.model import Model
from ..utils.logging import setup_logger
from ..core.attention import create_attention_mask

logger = setup_logger(__name__)

@dataclass
class AttentionPatterns:
    """Container for attention analysis results."""
    layer_index: int
    head_index: Optional[int]
    patterns: mx.array  # Shape: [batch_size, num_heads, seq_len, seq_len]
    tokens: List[str]
    scores: Optional[mx.array] = None  # Raw attention scores before softmax
    value_projections: Optional[mx.array] = None  # Value vectors
    key_projections: Optional[mx.array] = None  # Key vectors
    query_projections: Optional[mx.array] = None  # Query vectors
    aggregated_pattern: Optional[mx.array] = None  # Mean across heads if analyzing layer

class AttentionAnalyzer:
    """Analyzes attention patterns and mechanisms in transformer models."""
    
    def __init__(self, model: Model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self._validate_model()

    def _validate_model(self):
        """Ensure model has necessary attributes for analysis."""
        if not hasattr(self.model, "layers"):
            raise ValueError("Model must have 'layers' attribute")
        if not self.model.layers:
            raise ValueError("Model must have at least one layer")
        if not hasattr(self.model.layers[0], "self_attn"):
            raise ValueError("Model layers must have 'self_attn' attribute")

    def analyze_attention(
        self,
        text: str,
        layer_index: Optional[int] = None,
        head_index: Optional[int] = None,
        return_components: bool = False,
    ) -> Union[AttentionPatterns, List[AttentionPatterns]]:
        """Analyze attention patterns for given text.
        
        Args:
            text: Input text to analyze
            layer_index: Specific layer to analyze (None for all)
            head_index: Specific attention head to analyze (None for all)
            return_components: Whether to return query/key/value projections
            
        Returns:
            AttentionPatterns or list of AttentionPatterns
        """
        # Tokenize input
        tokens = self.tokenizer(text, return_tensors="np")
        input_ids = mx.array(tokens["input_ids"])
        token_texts = [self.tokenizer.decode([t]) for t in tokens["input_ids"][0]]

        # Forward pass with attention outputs
        attention_outputs = self._forward_with_attention(input_ids)
        
        # Analyze specific layer/head or all
        if layer_index is not None:
            return self._analyze_layer(
                attention_outputs,
                layer_index,
                head_index,
                token_texts,
                return_components
            )
        else:
            return [
                self._analyze_layer(
                    attention_outputs,
                    i,
                    head_index,
                    token_texts,
                    return_components
                )
                for i in range(len(self.model.layers))
            ]

    def _forward_with_attention(
        self,
        input_ids: mx.array,
    ) -> List[Dict[str, mx.array]]:
        """Perform forward pass collecting attention data."""
        attention_outputs = []
        hidden_states = self.model.embed_tokens(input_ids)
        
        # Create attention mask
        attention_mask = create_attention_mask(hidden_states, None)

        # Process each layer
        for layer in self.model.layers:
            # Get attention outputs
            layer_attn = layer.self_attn
            
            # Project hidden states
            queries = layer_attn.q_proj(hidden_states)
            keys = layer_attn.k_proj(hidden_states)
            values = layer_attn.v_proj(hidden_states)
            
            # Reshape for attention
            B, L, _ = hidden_states.shape
            H = layer_attn.n_heads
            D = layer_attn.head_dim
            
            queries = queries.reshape(B, L, H, D).transpose(0, 2, 1, 3)
            keys = keys.reshape(B, L, H, D).transpose(0, 2, 1, 3)
            values = values.reshape(B, L, H, D).transpose(0, 2, 1, 3)

            # Apply RoPE
            queries = layer_attn.rope(queries)
            keys = layer_attn.rope(keys)

            # Compute attention scores and patterns
            scale = 1 / mx.sqrt(D)
            scores = mx.matmul(queries, keys.transpose(0, 1, 3, 2)) * scale
            
            if attention_mask is not None:
                scores = scores + attention_mask
                
            patterns = mx.softmax(scores, axis=-1)
            
            # Compute attention output
            output = mx.matmul(patterns, values)
            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
            output = layer_attn.o_proj(output)
            
            # Store attention data
            attention_outputs.append({
                "scores": scores,
                "patterns": patterns,
                "queries": queries,
                "keys": keys,
                "values": values,
                "output": output
            })
            
            # Update hidden states
            hidden_states = hidden_states + output
            hidden_states = hidden_states + layer.mlp(
                layer.post_attention_layernorm(hidden_states)
            )

        return attention_outputs

    def _analyze_layer(
        self,
        attention_outputs: List[Dict[str, mx.array]],
        layer_index: int,
        head_index: Optional[int],
        tokens: List[str],
        return_components: bool,
    ) -> AttentionPatterns:
        """Analyze attention patterns for a specific layer."""
        layer_output = attention_outputs[layer_index]
        
        # Get attention patterns
        patterns = layer_output["patterns"]
        scores = layer_output["scores"] if return_components else None
        
        # Handle specific head analysis
        if head_index is not None:
            patterns = patterns[:, head_index:head_index+1, :, :]
            if scores is not None:
                scores = scores[:, head_index:head_index+1, :, :]
                
        # Get projections if requested
        value_projections = layer_output["values"] if return_components else None
        key_projections = layer_output["keys"] if return_components else None
        query_projections = layer_output["queries"] if return_components else None
        
        # Compute aggregated pattern if analyzing whole layer
        aggregated_pattern = None
        if head_index is None:
            aggregated_pattern = mx.mean(patterns, axis=1)
            
        return AttentionPatterns(
            layer_index=layer_index,
            head_index=head_index,
            patterns=patterns,
            tokens=tokens,
            scores=scores,
            value_projections=value_projections,
            key_projections=key_projections,
            query_projections=query_projections,
            aggregated_pattern=aggregated_pattern
        )

    def get_head_importance(
        self,
        text: str,
        layer_index: Optional[int] = None,
        method: str = "gradient",
    ) -> Dict[int, Dict[int, float]]:
        """Calculate attention head importance scores.
        
        Args:
            text: Input text
            layer_index: Specific layer to analyze (None for all)
            method: Importance scoring method ('gradient' or 'attention')
            
        Returns:
            Dictionary mapping layer indices to head importance scores
        """
        # Get attention patterns
        patterns = self.analyze_attention(text, layer_index)
        if not isinstance(patterns, list):
            patterns = [patterns]
            
        importance_scores = {}
        
        # Calculate importance for each layer
        for layer_patterns in patterns:
            layer_idx = layer_patterns.layer_index
            importance_scores[layer_idx] = {}
            
            # Calculate for each head
            num_heads = layer_patterns.patterns.shape[1]
            for head in range(num_heads):
                head_pattern = layer_patterns.patterns[:, head, :, :]
                
                if method == "gradient":
                    # Use gradient-based importance
                    score = self._calculate_gradient_importance(head_pattern)
                else:
                    # Use attention-based importance
                    score = self._calculate_attention_importance(head_pattern)
                    
                importance_scores[layer_idx][head] = float(score)
                
        return importance_scores

    def _calculate_gradient_importance(self, pattern: mx.array) -> float:
        """Calculate gradient-based importance score."""
        # Compute gradient of output with respect to attention pattern
        pattern_grad = mx.mean(mx.abs(mx.grad(pattern, pattern)))
        return float(pattern_grad)

    def _calculate_attention_importance(self, pattern: mx.array) -> float:
        """Calculate attention-based importance score."""
        # Use entropy of attention distribution
        entropy = -mx.sum(pattern * mx.log(pattern + 1e-10))
        return float(mx.mean(entropy))

    def get_attention_flow(
        self,
        text: str,
        start_token_index: int,
        end_token_index: int,
    ) -> Dict[str, mx.array]:
        """Analyze attention flow between specific tokens across layers."""
        patterns = self.analyze_attention(text)
        
        flow_stats = {
            "direct_attention": [],  # Direct attention weights
            "indirect_attention": [],  # Attention through intermediate tokens
            "total_attention": []  # Combined attention flow
        }
        
        # Process each layer
        for layer_pattern in patterns:
            # Get aggregated attention for layer
            if layer_pattern.aggregated_pattern is None:
                pattern = mx.mean(layer_pattern.patterns, axis=1)
            else:
                pattern = layer_pattern.aggregated_pattern
                
            # Get direct attention
            direct = pattern[:, start_token_index, end_token_index]
            flow_stats["direct_attention"].append(direct)
            
            # Calculate indirect attention through other tokens
            indirect = mx.zeros_like(direct)
            for k in range(pattern.shape[1]):
                if k != start_token_index and k != end_token_index:
                    path1 = pattern[:, start_token_index, k]
                    path2 = pattern[:, k, end_token_index]
                    indirect += path1 * path2
                    
            flow_stats["indirect_attention"].append(indirect)
            flow_stats["total_attention"].append(direct + indirect)
            
        return {k: mx.stack(v) for k, v in flow_stats.items()}

    def compare_attention_patterns(
        self,
        text1: str,
        text2: str,
        layer_index: Optional[int] = None,
        head_index: Optional[int] = None,
    ) -> Dict[str, float]:
        """Compare attention patterns between two inputs."""
        # Get attention patterns
        patterns1 = self.analyze_attention(text1, layer_index, head_index)
        patterns2 = self.analyze_attention(text2, layer_index, head_index)
        
        if isinstance(patterns1, list):
            return [
                self._compare_single_patterns(p1, p2)
                for p1, p2 in zip(patterns1, patterns2)
            ]
        else:
            return self._compare_single_patterns(patterns1, patterns2)

    def _compare_single_patterns(
        self,
        pattern1: AttentionPatterns,
        pattern2: AttentionPatterns,
    ) -> Dict[str, float]:
        """Compare two specific attention patterns."""
        # Get patterns (aggregated if needed)
        p1 = pattern1.aggregated_pattern if pattern1.aggregated_pattern is not None else pattern1.patterns
        p2 = pattern2.aggregated_pattern if pattern2.aggregated_pattern is not None else pattern2.patterns
        
        # Calculate various similarity metrics
        metrics = {
            "cosine_similarity": float(
                mx.mean(
                    mx.sum(p1 * p2, axis=-1) /
                    (mx.sqrt(mx.sum(p1 * p1, axis=-1)) * mx.sqrt(mx.sum(p2 * p2, axis=-1)))
                )
            ),
            "l2_distance": float(
                mx.mean(mx.sqrt(mx.sum((p1 - p2) ** 2, axis=-1)))
            ),
            "kl_divergence": float(
                mx.mean(mx.sum(p1 * mx.log(p1 / (p2 + 1e-10) + 1e-10), axis=-1))
            )
        }
        
        return metrics