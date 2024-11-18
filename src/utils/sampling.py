# utils/sampling.py

from typing import Optional, List, Tuple
import mlx.core as mx

def prepare_logits(
    logits: mx.array,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    repetition_context: Optional[List[int]] = None,
) -> mx.array:
    """Prepare logits for sampling with various strategies."""
    # Temperature scaling
    if temperature == 0:
        # Greedy selection
        return mx.argmax(logits, axis=-1, keepdims=True)
        
    logits = logits / temperature

    # Apply repetition penalty
    if repetition_penalty and repetition_context:
        logits = apply_repetition_penalty(logits, repetition_context, repetition_penalty)

    # Top-K filtering
    if top_k is not None:
        v, _ = mx.top_k(logits, min(top_k, logits.shape[-1]))
        logits = mx.where(logits < v[:, [-1]], -float('inf'), logits)

    # Top-p (nucleus) filtering
    if top_p is not None:
        sorted_logits = mx.sort(logits, axis=-1, descending=True)
        cumulative_probs = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Shift the indices to keep first token above threshold
        sorted_indices_to_remove = mx.roll(sorted_indices_to_remove, 1, axis=-1)
        sorted_indices_to_remove = sorted_indices_to_remove.at[..., 0].set(False)
        
        indices_to_remove = mx.take_along_axis(
            sorted_indices_to_remove,
            mx.argsort(-logits, axis=-1),
            axis=-1
        )
        logits = mx.where(indices_to_remove, -float('inf'), logits)

    return logits

def sample_token(
    logits: mx.array,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    repetition_context: Optional[List[int]] = None,
) -> Tuple[mx.array, mx.array]:
    """Sample next token from prepared logits."""
    prepared_logits = prepare_logits(
        logits,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        repetition_context
    )
    
    if temperature == 0:
        next_token = prepared_logits
    else:
        probs = mx.softmax(prepared_logits, axis=-1)
        next_token = mx.random.categorical(prepared_logits)

    return next_token, probs