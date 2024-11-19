from typing import Dict, List, Optional, Tuple
import mlx.core as mx

class BatchedKVCache:
    """Enhanced KV cache with batch support and memory management."""
    
    def __init__(self, head_dim: int, n_kv_heads: int, batch_size: int = 1):
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.batch_size = batch_size
        self.keys = None
        self.values = None
        self.offset = 0
        self.step = 256
        
    def update_and_fetch(
        self, 
        keys: mx.array, 
        values: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """Update cache and return current state."""
        seq_len = keys.shape[2]  # Get sequence length from input
        
        # Handle dynamic resizing
        if self.keys is None or (self.offset + seq_len) > self.keys.shape[2]:
            n_steps = (self.step + seq_len - 1) // self.step
            shape = (
                self.batch_size,
                self.n_kv_heads,
                n_steps * self.step,
                self.head_dim,
            )
            new_k = mx.zeros(shape, keys.dtype)
            new_v = mx.zeros(shape, values.dtype)
            
            if self.keys is not None:
                new_k[..., :self.offset, :] = self.keys[..., :self.offset, :]
                new_v[..., :self.offset, :] = self.values[..., :self.offset, :]
            
            self.keys = new_k
            self.values = new_v

        # Update cache with new values
        self.keys[..., self.offset:self.offset + seq_len, :] = keys
        self.values[..., self.offset:self.offset + seq_len, :] = values
        
        # Update offset after storing new values
        self.offset += seq_len
        
        return self.keys[..., :self.offset, :], self.values[..., :self.offset, :]
    
    def get_sequence_state(self, sequence_id: str) -> Optional[Dict]:
        """Get cached state for a specific sequence."""
        return self.sequence_states.get(sequence_id) 