# analysis/layers.py

import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import mlx.core as mx
from sklearn.decomposition import PCA

from ..core.model import Model
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

@dataclass
class LayerActivations:
    """Container for layer activation analysis results."""
    layer_index: int
    hidden_states: mx.array
    mlp_activations: Optional[mx.array] = None
    neuron_activations: Optional[mx.array] = None
    gate_values: Optional[mx.array] = None
    up_project: Optional[mx.array] = None
    down_project: Optional[mx.array] = None
    pre_attention: Optional[mx.array] = None
    post_attention: Optional[mx.array] = None
    tokens: List[str] = None

@dataclass
class LayerAnalysis:
    """Results from analyzing layer behavior."""
    layer_index: int
    neuron_importance: Dict[int, float]
    activation_statistics: Dict[str, Dict[str, float]]
    pca_components: Optional[mx.array] = None
    explained_variance: Optional[mx.array] = None

class LayerAnalyzer:
    """Analyzes internal layer representations and behaviors."""

    def __init__(self, model: Model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self._validate_model()

    def _validate_model(self):
        """Ensure model has necessary components."""
        if not hasattr(self.model, "layers"):
            raise ValueError("Model must have 'layers' attribute")
        for i, layer in enumerate(self.model.layers):
            if not hasattr(layer, "mlp"):
                raise ValueError(f"Layer {i} missing MLP component")

    def get_layer_activations(
        self,
        texts: Union[str, List[str]],
        layer_index: Optional[int] = None,
        return_components: bool = False,
        batch_size: int = 32
    ) -> Union[LayerActivations, List[LayerActivations]]:
        """Get activation patterns for specified layer(s) with batch support."""
        if isinstance(texts, str):
            texts = [texts]
        
        from ..inference.batch_manager import BatchManager, BatchConfig
        
        # Initialize batch manager
        config = BatchConfig(max_batch_size=batch_size)
        batch_manager = BatchManager(self.model, self.tokenizer, config)
        
        results = []
        for batch in batch_manager.create_batches(texts):
            # Process batch
            batch_activations = self._forward_with_activations(
                batch['input_ids'],
                return_components
            )
            
            # Process results for each sequence in batch
            for seq_idx in range(len(batch['sequence_ids'])):
                if layer_index is not None:
                    # Single layer for one sequence
                    results.append(self._process_layer_activations(
                        batch_activations[layer_index],
                        layer_index,
                        [self.tokenizer.decode([t]) for t in batch['input_ids'][seq_idx]],
                        return_components
                    ))
                else:
                    # All layers for one sequence
                    results.append([
                        self._process_layer_activations(
                            act,
                            i,
                            [self.tokenizer.decode([t]) for t in batch['input_ids'][seq_idx]],
                            return_components
                        )
                        for i, act in enumerate(batch_activations)
                    ])
        
        # Return single result for single input
        if len(texts) == 1:
            return results[0]
        return results

    def _forward_with_activations(
        self,
        input_ids: mx.array,
        return_components: bool,
    ) -> List[Dict[str, mx.array]]:
        """Forward pass collecting layer activations."""
        activations = []
        hidden_states = self.model.embed_tokens(input_ids)

        for layer in self.model.layers:
            layer_activations = {}

            # Pre-attention processing
            pre_attention = layer.input_layernorm(hidden_states)
            if return_components:
                layer_activations["pre_attention"] = pre_attention

            # Attention
            attention_output = layer.self_attn(pre_attention)
            post_attention = hidden_states + attention_output
            if return_components:
                layer_activations["post_attention"] = post_attention

            # MLP components
            mlp_input = layer.post_attention_layernorm(post_attention)
            gate_values = layer.mlp.gate_proj(mlp_input)
            up_values = layer.mlp.up_proj(mlp_input)

            # Compute activations
            neuron_values = mx.silu(gate_values) * up_values
            mlp_output = layer.mlp.down_proj(neuron_values)

            # Store activations
            layer_activations.update({
                "hidden_states": hidden_states,
                "mlp_activations": mlp_output,
                "neuron_activations": neuron_values,
            })

            if return_components:
                layer_activations.update({
                    "gate_values": gate_values,
                    "up_project": up_values,
                    "down_project": mlp_output,
                })

            activations.append(layer_activations)

            # Update hidden states
            hidden_states = post_attention + mlp_output

        return activations

    def _process_layer_activations(
        self,
        activations: Dict[str, mx.array],
        layer_index: int,
        tokens: List[str],
        return_components: bool,
    ) -> LayerActivations:
        """Process raw activations into LayerActivations object."""
        return LayerActivations(
            layer_index=layer_index,
            hidden_states=activations["hidden_states"],
            mlp_activations=activations["mlp_activations"],
            neuron_activations=activations["neuron_activations"],
            gate_values=activations.get("gate_values") if return_components else None,
            up_project=activations.get("up_project") if return_components else None,
            down_project=activations.get("down_project") if return_components else None,
            pre_attention=activations.get("pre_attention") if return_components else None,
            post_attention=activations.get("post_attention") if return_components else None,
            tokens=tokens,
        )

    def analyze_layer(
        self,
        text: str,
        layer_index: int,
        n_components: int = 10,
    ) -> LayerAnalysis:
        """Comprehensive analysis of layer behavior."""
        # Get activations
        activations = self.get_layer_activations(text, layer_index, return_components=True)

        # Analyze neuron importance
        neuron_importance = self._calculate_neuron_importance(activations)

        # Calculate activation statistics
        activation_stats = self._calculate_activation_statistics(activations)

        # Perform PCA on hidden states
        hidden_states = activations.hidden_states.reshape(-1, activations.hidden_states.shape[-1])
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(hidden_states)

        return LayerAnalysis(
            layer_index=layer_index,
            neuron_importance=neuron_importance,
            activation_statistics=activation_stats,
            pca_components=mx.array(pca_result),
            explained_variance=mx.array(pca.explained_variance_ratio_)
        )

    def _calculate_neuron_importance(
        self,
        activations: LayerActivations,
    ) -> Dict[int, float]:
        """Calculate importance scores for individual neurons."""
        neuron_acts = activations.neuron_activations
        
        # Calculate importance based on activation magnitude and sparsity
        mean_activation = mx.mean(mx.abs(neuron_acts), axis=(0, 1))
        sparsity = mx.mean(neuron_acts == 0.0, axis=(0, 1))
        
        # Combine metrics into importance score
        importance = mean_activation * (1 - sparsity)
        
        return {i: float(score) for i, score in enumerate(importance)}

    def _calculate_activation_statistics(
        self,
        activations: LayerActivations,
    ) -> Dict[str, Dict[str, float]]:
        """Calculate statistical measures of layer activations."""
        stats = {}
        
        # Analyze different activation components
        components = {
            "hidden_states": activations.hidden_states,
            "mlp_activations": activations.mlp_activations,
            "neuron_activations": activations.neuron_activations,
        }
        
        for name, acts in components.items():
            stats[name] = {
                "mean": float(mx.mean(acts)),
                "std": float(mx.std(acts)),
                "max": float(mx.max(acts)),
                "min": float(mx.min(acts)),
                "sparsity": float(mx.mean(acts == 0.0)),
            }
            
        return stats

    def compare_layer_behaviors(
        self,
        text1: str,
        text2: str,
        layer_index: Optional[int] = None,
    ) -> Dict[str, float]:
        """Compare layer behaviors between two inputs."""
        # Get activations for both inputs
        acts1 = self.get_layer_activations(text1, layer_index)
        acts2 = self.get_layer_activations(text2, layer_index)
        
        if isinstance(acts1, list):
            return [
                self._compare_layer_activations(a1, a2)
                for a1, a2 in zip(acts1, acts2)
            ]
        else:
            return self._compare_layer_activations(acts1, acts2)

    def _compare_layer_activations(
        self,
        acts1: LayerActivations,
        acts2: LayerActivations,
    ) -> Dict[str, float]:
        """Compare activation patterns between two layer states."""
        metrics = {}
        
        # Compare hidden states
        h1 = acts1.hidden_states.reshape(-1, acts1.hidden_states.shape[-1])
        h2 = acts2.hidden_states.reshape(-1, acts2.hidden_states.shape[-1])
        
        # Cosine similarity
        h1_norm = mx.sqrt(mx.sum(h1 * h1, axis=-1, keepdims=True))
        h2_norm = mx.sqrt(mx.sum(h2 * h2, axis=-1, keepdims=True))
        cos_sim = mx.mean(mx.sum(h1 * h2, axis=-1) / (h1_norm * h2_norm))
        
        # L2 distance
        l2_dist = mx.mean(mx.sqrt(mx.sum((h1 - h2) ** 2, axis=-1)))
        
        # Activation pattern correlation
        if acts1.neuron_activations is not None and acts2.neuron_activations is not None:
            n1 = acts1.neuron_activations.reshape(-1, acts1.neuron_activations.shape[-1])
            n2 = acts2.neuron_activations.reshape(-1, acts2.neuron_activations.shape[-1])
            
            # Correlation coefficient
            n1_centered = n1 - mx.mean(n1, axis=0, keepdims=True)
            n2_centered = n2 - mx.mean(n2, axis=0, keepdims=True)
            corr = mx.mean(
                mx.sum(n1_centered * n2_centered, axis=0) /
                (mx.sqrt(mx.sum(n1_centered ** 2, axis=0)) * mx.sqrt(mx.sum(n2_centered ** 2, axis=0)))
            )
            
            metrics["neuron_correlation"] = float(corr)
        
        metrics.update({
            "hidden_cosine_similarity": float(cos_sim),
            "hidden_l2_distance": float(l2_dist),
        })
        
        return metrics

    def get_neuron_patterns(
        self,
        texts: List[str],
        layer_index: int,
        top_k_neurons: int = 10,
    ) -> Dict[int, Dict[str, mx.array]]:
        """Analyze activation patterns of individual neurons."""
        all_activations = []
        
        # Collect activations for all inputs
        for text in texts:
            acts = self.get_layer_activations(text, layer_index)
            all_activations.append(acts.neuron_activations)
            
        # Stack activations
        stacked_acts = mx.concatenate(all_activations, axis=0)
        
        # Find most active neurons
        mean_acts = mx.mean(mx.abs(stacked_acts), axis=(0, 1))
        top_neurons = mx.argsort(-mean_acts)[:top_k_neurons]
        
        patterns = {}
        for neuron_idx in top_neurons:
            neuron_acts = stacked_acts[..., neuron_idx]
            
            patterns[int(neuron_idx)] = {
                "activations": neuron_acts,
                "mean": float(mx.mean(neuron_acts)),
                "std": float(mx.std(neuron_acts)),
                "max_activation": float(mx.max(neuron_acts)),
                "sparsity": float(mx.mean(neuron_acts == 0.0)),
            }
            
        return patterns