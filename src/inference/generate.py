# inference/generate.py

import logging
from typing import List, Optional, Union, Dict, Any, Tuple, Generator
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from transformers import PreTrainedTokenizer

from src.core.model import Model, BatchedKVCache
from src.core.attention import create_attention_mask

logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 1.0
    repetition_penalty: Optional[float] = None
    repetition_context_size: Optional[int] = 20
    logit_bias: Optional[Dict[int, float]] = None
    return_hidden_states: bool = False

def top_p_sampling(
    logits: mx.array,
    temperature: float,
    top_p: float = 1.0,
) -> mx.array:
    """Nucleus sampling with temperature."""
    # Temperature scaling
    if temperature == 0:
        return mx.argmax(logits, axis=-1)

    # Apply temperature
    logits = logits * (1 / temperature)
    
    # Calculate probabilities
    probs = mx.softmax(logits, axis=-1)
    
    if top_p < 1.0:
        # Sort probabilities descending
        sorted_indices = mx.argsort(-probs, axis=-1)
        sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
        
        # Compute cumulative probabilities
        cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
        
        # Create probability mask
        sorted_mask = cumulative_probs <= top_p
        sorted_mask = mx.logical_or(
            sorted_mask,
            mx.arange(sorted_mask.shape[-1]) == 0
        )
        
        # Filter probabilities
        sorted_probs = mx.where(sorted_mask, sorted_probs, 0.0)
        sorted_probs = sorted_probs / mx.sum(sorted_probs, axis=-1, keepdims=True)
        
        # Sample from filtered distribution
        sorted_indices = mx.random.categorical(mx.log(sorted_probs))
        indices = mx.take_along_axis(sorted_indices, sorted_indices, axis=-1)
    else:
        # Direct sampling
        indices = mx.random.categorical(logits)
    
    return indices

def apply_repetition_penalty(
    logits: mx.array,
    generated_tokens: List[int],
    penalty: float,
) -> mx.array:
    """Apply repetition penalty to logits."""
    if len(generated_tokens) == 0:
        return logits
        
    indices = mx.array(generated_tokens)
    selected_logits = logits[:, indices]
    
    selected_logits = mx.where(
        selected_logits < 0,
        selected_logits * penalty,
        selected_logits / penalty
    )
    
    logits[:, indices] = selected_logits
    return logits

def generate(
    model: Model,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompts: List[str],
    generation_config: Optional[GenerationConfig] = None,
    emphasis_config: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Generator[Tuple[List[str], Optional[Dict]], None, None]:
    """Generate text from prompts."""
    if generation_config is None:
        generation_config = GenerationConfig()

    # Prepare tokenizer
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    tokenizer._tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer._tokenizer.pad_token = tokenizer.eos_token
        tokenizer._tokenizer.pad_token_id = tokenizer.eos_token_id

    # Tokenize prompts
    input_ids = mx.array(
        tokenizer._tokenizer(
            prompts,
            padding=True,
            return_tensors='np'
        )['input_ids']
    )

    # Apply emphasis configuration if provided
    if emphasis_config:
        model.set_emphasis_config(emphasis_config)

    # Initialize generation
    generated_tokens = []
    response_texts = [""] * len(prompts)
    
    # Setup KV cache
    kv_cache = [
        BatchedKVCache(model.head_dim, model.n_kv_heads, input_ids.shape[0])
        for _ in range(len(model.layers))
    ]

    # Generate tokens
    for step in range(generation_config.max_tokens):
        # Forward pass
        if step == 0:
            logits, hidden_states = model(
                input_ids,
                cache=kv_cache,
                return_hidden_states=generation_config.return_hidden_states
            )
            logits = logits[:, -1, :]
        else:
            next_logits, hidden_states = model(
                next_tokens,
                cache=kv_cache,
                return_hidden_states=generation_config.return_hidden_states
            )
            logits = next_logits[:, -1, :]

        # Apply repetition penalty if configured
        if generation_config.repetition_penalty:
            logits = apply_repetition_penalty(
                logits,
                generated_tokens[-generation_config.repetition_context_size:],
                generation_config.repetition_penalty
            )

        # Apply logit bias if provided
        if generation_config.logit_bias:
            for token_id, bias in generation_config.logit_bias.items():
                logits[:, token_id] += bias

        # Sample next tokens
        next_tokens = top_p_sampling(
            logits,
            generation_config.temperature,
            generation_config.top_p
        )

        # Decode and update responses
        for i, token in enumerate(next_tokens.tolist()):
            generated_tokens.append(token)
            token_text = tokenizer.decode([token])
            response_texts[i] += token_text

            # Check for EOS
            if token == tokenizer.eos_token_id:
                response_texts[i] = response_texts[i].split(tokenizer.eos_token)[0]

        # Prepare output
        output = {
            'responses': response_texts,
            'hidden_states': hidden_states if generation_config.return_hidden_states else None
        }

        yield response_texts, output

        # Check if all responses are complete
        if all(tokenizer.eos_token_id in resp for resp in generated_tokens):
            break

        # Prepare for next iteration
        next_tokens = mx.reshape(next_tokens, (len(prompts), 1))

    if verbose:
        for prompt, response in zip(prompts, response_texts):
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Response: {response}")
            logger.info("-" * 50)