from typing import Dict, List, Optional, Union
import mlx.core as mx
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class EmbeddingAnalyzer:
    """Analyzes token embeddings and their relationships."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def extract_embeddings(
        self,
        text: Union[str, List[str]],
        layer: Optional[int] = None,
        aggregate: str = "mean"
    ) -> mx.array:
        """Extract embeddings from specified layer."""
        if isinstance(text, str):
            text = [text]
            
        # Tokenize input
        tokens = self.tokenizer(text, return_tensors="np", padding=True)
        input_ids = mx.array(tokens["input_ids"])
        
        # Get model outputs
        outputs = self.model(
            input_ids,
            return_hidden_states=True
        )
        
        # Extract hidden states
        hidden_states = outputs[1]
        
        # Select layer
        if layer is not None:
            embeddings = hidden_states[layer]
        else:
            embeddings = hidden_states[-1]
            
        # Aggregate embeddings
        if aggregate == "mean":
            embeddings = mx.mean(embeddings, axis=1)
        elif aggregate == "max":
            embeddings = mx.max(embeddings, axis=1)
        
        return embeddings
    
    def reduce_dimensions(
        self,
        embeddings: mx.array,
        method: str = "pca",
        n_components: int = 2
    ) -> np.ndarray:
        """Reduce embedding dimensions for visualization."""
        # Convert to numpy for sklearn
        embeddings_np = embeddings.numpy()
        
        if method == "pca":
            reducer = PCA(n_components=n_components)
        elif method == "tsne":
            reducer = TSNE(n_components=n_components)
        else:
            raise ValueError(f"Unsupported reduction method: {method}")
            
        return reducer.fit_transform(embeddings_np) 