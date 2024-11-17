import mlx.core as mx
from typing import List, Optional
from dataclasses import dataclass
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

@dataclass
class GenerationConfig:
    max_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    stop_tokens: List[int] = None

def generate_with_emphasis(
    model,
    tokenizer,
    prompts: List[str],
    config: GenerationConfig,
    emphasis_config: dict = None,
    verbose: bool = False
) -> List[str]:
    """
    Generate text with emphasis/ablation controls
    """
    if emphasis_config:
        from .emphasis import (
            apply_layer_emphasis,
            apply_head_emphasis,
            apply_neuron_emphasis
        )
        
        if 'layers' in emphasis_config:
            apply_layer_emphasis(model, emphasis_config['layers'])
        if 'heads' in emphasis_config:
            apply_head_emphasis(model, emphasis_config['heads'])
        if 'neurons' in emphasis_config:
            apply_neuron_emphasis(model, emphasis_config['neurons'])

    # Format prompts
    prompts_fm = [[{"role": "user", "content": p}] for p in prompts]
    prompts_fm = [
        tokenizer.apply_chat_template(p, add_generation_prompt=True, tokenize=False)
        for p in prompts_fm
    ]

    # Tokenize
    encoded = tokenizer(prompts_fm, padding=True, return_tensors="np")
    input_ids = mx.array(encoded['input_ids'])

    # Generate
    outputs = []
    for tokens in generate_tokens(model, input_ids, config):
        outputs.append(tokens)
        if len(outputs) >= config.max_tokens:
            break

    output_ids = mx.concatenate(outputs, axis=-1)
    
    # Decode
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
