from typing import List, Optional, Dict, Any
import mlx.core as mx
from .generator import GenerationConfig, generate_tokens
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

def generate_with_emphasis(
    model,
    tokenizer,
    prompts: List[str],
    max_tokens: int = 100,
    emphasis_config: Optional[Dict[str, Any]] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: Optional[int] = None,
    verbose: bool = False
) -> List[str]:
    """Generate text with emphasis/ablation controls"""
    
    # Apply emphasis if provided
    if emphasis_config:
        model.set_emphasis_config(emphasis_config)

    # Configure generation
    config = GenerationConfig(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k
    )

    # Format and tokenize prompts
    formatted_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True
        ) for p in prompts
    ]
    
    input_ids = mx.array(tokenizer(
        formatted_prompts,
        padding=True,
        return_tensors="np"
    )['input_ids'])

    # Generate
    outputs = []
    for tokens in generate_tokens(model, input_ids, config):
        outputs.append(tokens)
        if len(outputs) >= config.max_tokens:
            break

    # Decode responses
    output_ids = mx.concatenate(outputs, axis=-1)
    responses = []
    for ids in output_ids:
        text = tokenizer.decode(ids.tolist())
        text = text.split(tokenizer.eos_token)[0]
        responses.append(text)

    if verbose:
        for prompt, response in zip(prompts, responses):
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Response: {response}")
            logger.info("-" * 80)

    return responses
