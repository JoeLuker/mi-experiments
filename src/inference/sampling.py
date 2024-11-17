import mlx.core as mx

def top_p_sampling(logits: mx.array, top_p: float = 0.9) -> mx.array:
    """
    Nucleus (top-p) sampling implementation
    """
    probs = mx.softmax(logits, axis=-1)
    sorted_probs = mx.sort(probs, axis=-1, descending=True)
    cumsum_probs = mx.cumsum(sorted_probs, axis=-1)
    mask = cumsum_probs <= top_p
    
    # Keep at least one token
    mask = mx.logical_or(mask, mx.arange(mask.shape[-1]) == 0)
    
    sorted_mask = mx.sort_permutation(probs, axis=-1, descending=True)
    reverse_mask = mx.sort_permutation(sorted_mask, axis=-1)
    
    masked_probs = mx.where(
        mask[reverse_mask],
        probs,
        mx.zeros_like(probs)
    )
    
    # Renormalize
    masked_probs = masked_probs / mx.sum(masked_probs, axis=-1, keepdims=True)
    return masked_probs 