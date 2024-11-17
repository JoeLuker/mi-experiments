from dataclasses import dataclass
from typing import Optional, List, Tuple
import mlx.core as mx

@dataclass
class BatchedKVCache:
    """Efficient key-value cache for batch processing"""
    keys: List[mx.array]
    values: List[mx.array]
    
    @classmethod
    def create(cls, num_layers: int, batch_size: int, num_heads: int, head_dim: int):
        return cls(
            keys=[[] for _ in range(num_layers)],
            values=[[] for _ in range(num_layers)]
        )
    
    def update(self, layer_idx: int, key: mx.array, value: mx.array):
        self.keys[layer_idx].append(key)
        self.values[layer_idx].append(value)
    
    def get_layer(self, layer_idx: int) -> Tuple[Optional[mx.array], Optional[mx.array]]:
        if not self.keys[layer_idx]:
            return None, None
        return (
            mx.concatenate(self.keys[layer_idx], axis=1),
            mx.concatenate(self.values[layer_idx], axis=1)
        ) 