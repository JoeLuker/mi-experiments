import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Optional, Union
import numpy as np
import mlx.core as mx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class VisualizationTools:
    """Tools for visualizing model analysis results."""
    
    @staticmethod
    def plot_attention_patterns(
        attention_weights: mx.array,
        tokens: List[str],
        layer_idx: Optional[int] = None,
        head_idx: Optional[int] = None
    ) -> go.Figure:
        """Plot attention pattern heatmap."""
        weights = attention_weights.numpy()
        
        fig = go.Figure(data=go.Heatmap(
            z=weights,
            x=tokens,
            y=tokens,
            colorscale="Viridis",
            colorbar=dict(title="Attention Weight")
        ))
        
        title = "Attention Patterns"
        if layer_idx is not None:
            title += f" - Layer {layer_idx}"
        if head_idx is not None:
            title += f" - Head {head_idx}"
            
        fig.update_layout(
            title=title,
            xaxis_title="Key Tokens",
            yaxis_title="Query Tokens",
            width=800,
            height=600
        )
        
        return fig
    
    @staticmethod
    def plot_token_embeddings(
        embeddings: mx.array,
        tokens: List[str],
        method: str = "pca",
        n_components: int = 2
    ) -> go.Figure:
        """Visualize token embeddings using dimensionality reduction."""
        # Convert to numpy for sklearn
        emb_np = embeddings.numpy()
        
        # Reduce dimensions
        if method == "pca":
            reducer = PCA(n_components=n_components)
        elif method == "tsne":
            reducer = TSNE(n_components=n_components)
        else:
            raise ValueError(f"Unsupported reduction method: {method}")
            
        reduced = reducer.fit_transform(emb_np)
        
        # Create scatter plot
        if n_components == 2:
            fig = go.Figure(data=go.Scatter(
                x=reduced[:, 0],
                y=reduced[:, 1],
                mode='text+markers',
                text=tokens,
                textposition="top center",
                marker=dict(size=10)
            ))
            
            fig.update_layout(
                title=f"Token Embeddings ({method.upper()})",
                xaxis_title=f"{method.upper()} 1",
                yaxis_title=f"{method.upper()} 2",
                width=800,
                height=600
            )
        else:
            fig = go.Figure(data=go.Scatter3d(
                x=reduced[:, 0],
                y=reduced[:, 1],
                z=reduced[:, 2],
                mode='text+markers',
                text=tokens,
                marker=dict(size=5)
            ))
            
            fig.update_layout(
                title=f"Token Embeddings ({method.upper()})",
                scene=dict(
                    xaxis_title=f"{method.upper()} 1",
                    yaxis_title=f"{method.upper()} 2",
                    zaxis_title=f"{method.upper()} 3"
                ),
                width=800,
                height=800
            )
            
        return fig
    
    @staticmethod
    def plot_layer_activations(
        activations: mx.array,
        layer_idx: int,
        tokens: Optional[List[str]] = None,
        top_k_neurons: Optional[int] = None
    ) -> go.Figure:
        """Visualize layer activation patterns."""
        acts = activations.numpy()
        
        if tokens is None:
            tokens = [f"Token {i}" for i in range(acts.shape[1])]
            
        if top_k_neurons is not None:
            # Select top-k most active neurons
            mean_acts = np.mean(np.abs(acts), axis=(0, 1))
            top_indices = np.argsort(-mean_acts)[:top_k_neurons]
            acts = acts[:, :, top_indices]
            neuron_labels = [f"Neuron {i}" for i in top_indices]
        else:
            neuron_labels = [f"Neuron {i}" for i in range(acts.shape[-1])]
        
        fig = go.Figure(data=go.Heatmap(
            z=acts.squeeze(),
            x=tokens,
            y=neuron_labels,
            colorscale="RdBu",
            colorbar=dict(title="Activation")
        ))
        
        fig.update_layout(
            title=f"Layer {layer_idx} Activations",
            xaxis_title="Tokens",
            yaxis_title="Neurons",
            width=1000,
            height=600
        )
        
        return fig
