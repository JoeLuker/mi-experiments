from typing import List, Dict, Optional, Generator
import mlx.core as mx
from dataclasses import dataclass

from ..core.cache import BatchedKVCache
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    max_batch_size: int = 32
    max_sequence_length: int = 2048
    max_batch_tokens: int = 1024  # For dynamic batching
    pad_token_id: int = 0
    dynamic_batching: bool = True

class BatchManager:
    """Manages batched operations and sequence processing."""
    
    def __init__(
        self,
        model,
        tokenizer,
        config: Optional[BatchConfig] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or BatchConfig()
        self.current_batch = []
        self.sequence_map = {}  # Maps sequence IDs to batch indices
        
    def create_batches(
        self,
        sequences: List[str],
        return_attention: bool = False
    ) -> Generator[Dict[str, mx.array], None, None]:
        """Create optimized batches from input sequences."""
        # Tokenize all sequences
        tokenized = self.tokenizer(
            sequences,
            padding=True,
            return_tensors="np"
        )
        input_ids = mx.array(tokenized["input_ids"])
        
        if self.config.dynamic_batching:
            yield from self._create_dynamic_batches(input_ids, return_attention)
        else:
            yield from self._create_fixed_batches(input_ids, return_attention)
    
    def _create_fixed_batches(
        self,
        input_ids: mx.array,
        return_attention: bool
    ) -> Generator[Dict[str, mx.array], None, None]:
        """Create fixed-size batches."""
        for i in range(0, len(input_ids), self.config.max_batch_size):
            batch_ids = input_ids[i:i + self.config.max_batch_size]
            
            if batch_ids.shape[1] > self.config.max_sequence_length:
                batch_ids = batch_ids[:, :self.config.max_sequence_length]
                
            attention_mask = mx.ones(batch_ids.shape[:2]) if return_attention else None
            
            yield {
                'input_ids': batch_ids,
                'attention_mask': attention_mask,
                'sequence_ids': [f"seq_{j}" for j in range(i, i + len(batch_ids))]
            }
    
    def _create_dynamic_batches(
        self,
        input_ids: mx.array,
        return_attention: bool
    ) -> Generator[Dict[str, mx.array], None, None]:
        """Create dynamically-sized batches based on sequence lengths."""
        current_batch = []
        current_tokens = 0
        batch_start_idx = 0
        
        for i, seq in enumerate(input_ids):
            seq_len = len(seq)
            
            # Check if adding this sequence would exceed token limit
            if current_tokens + seq_len > self.config.max_batch_tokens and current_batch:
                batch_ids = mx.stack(current_batch)
                if batch_ids.shape[1] > self.config.max_sequence_length:
                    batch_ids = batch_ids[:, :self.config.max_sequence_length]
                    
                attention_mask = mx.ones(batch_ids.shape[:2]) if return_attention else None
                
                yield {
                    'input_ids': batch_ids,
                    'attention_mask': attention_mask,
                    'sequence_ids': [f"seq_{j}" for j in range(batch_start_idx, batch_start_idx + len(batch_ids))]
                }
                
                current_batch = []
                current_tokens = 0
                batch_start_idx = i
            
            current_batch.append(seq)
            current_tokens += seq_len
            
            # Yield batch if size limit reached
            if len(current_batch) >= self.config.max_batch_size:
                batch_ids = mx.stack(current_batch)
                if batch_ids.shape[1] > self.config.max_sequence_length:
                    batch_ids = batch_ids[:, :self.config.max_sequence_length]
                    
                attention_mask = mx.ones(batch_ids.shape[:2]) if return_attention else None
                
                yield {
                    'input_ids': batch_ids,
                    'attention_mask': attention_mask,
                    'sequence_ids': [f"seq_{j}" for j in range(batch_start_idx, batch_start_idx + len(batch_ids))]
                }
                
                current_batch = []
                current_tokens = 0
                batch_start_idx = i + 1
        
        # Yield any remaining sequences
        if current_batch:
            batch_ids = mx.stack(current_batch)
            if batch_ids.shape[1] > self.config.max_sequence_length:
                batch_ids = batch_ids[:, :self.config.max_sequence_length]
                
            attention_mask = mx.ones(batch_ids.shape[:2]) if return_attention else None
            
            yield {
                'input_ids': batch_ids,
                'attention_mask': attention_mask,
                'sequence_ids': [f"seq_{j}" for j in range(batch_start_idx, batch_start_idx + len(batch_ids))]
            }