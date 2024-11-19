from typing import List, Union, Optional, Tuple, Generator

import mlx.core as mx
import mlx.nn as nn

from mi_experiments.core.cache import BatchedKVCache
from mi_experiments.core.config import ControlVector
from mi_experiments.core.model import Model
from mi_experiments.core.rope import TokenizerWrapper, PreTrainedTokenizer
from mi_experiments.core.attention import create_attention_mask



def batch_generate_with_control(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompts: List[str],
    control_vector: ControlVector,
    control_strength: float = 1.0,
    max_tokens: int = 100,
    verbose: bool = False,
    format_prompts: bool = True,
    **kwargs,
) -> List[str]:
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    if verbose:
        print("=" * 10)
    
    if format_prompts:
        formatted_prompts = [
            tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
            for prompt in prompts
        ]
    else:
        formatted_prompts = prompts

    tokenizer._tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer._tokenizer.pad_token = tokenizer.eos_token
        tokenizer._tokenizer.pad_token_id = tokenizer.eos_token_id

    # Combined tokenization step
    prompts_toks = mx.array(tokenizer._tokenizer(formatted_prompts, padding=True)['input_ids'])
    start_time = time.perf_counter()

    def apply_control_vector(hidden_states, layer_index):
        if layer_index in control_vector.directions:
            direction = control_vector.directions[layer_index]
            control = mx.array(direction).reshape(1, 1, -1)
            hidden_states = hidden_states + control * control_strength
        return hidden_states
    
    def generate_step_with_control(
        prompts: mx.array,
        model: nn.Module,
        temp: float = 0.0,
        top_p: float = 1.0,
        return_hidden_states: bool = False,
        control_layers: List[int] = []
    ) -> Generator[Tuple[mx.array, mx.array, Optional[List[mx.array]]], None, None]:
        
        def sample(logits: mx.array) -> Tuple[mx.array, float]:
            softmax_logits = mx.softmax(logits, axis=-1)

            if temp == 0:
                tokens = mx.argmax(logits, axis=-1, keepdims=True)
            else:
                if top_p > 0 and top_p < 1.0:
                    tokens = top_p_sampling(logits, top_p, temp)
                else:
                    scaled_logits = logits * (1 / temp)
                    tokens = mx.random.categorical(logits * (1 / temp), axis=-1)
                    if scaled_logits.ndim > 1:
                        tokens = mx.expand_dims(tokens, axis=-1)

            probs = softmax_logits[0, tokens]
            return tokens, probs

        y = prompts
        kv_heads = (
            [model.n_kv_heads] * len(model.layers)
            if isinstance(model.n_kv_heads, int)
            else model.n_kv_heads
        )

        cache = [BatchedKVCache(model.head_dim, n, y.shape[0]) for n in kv_heads]

        def _step(y):
            nonlocal cache

            # call model

            hidden_states = model.model.embed_tokens(y)
            attention_mask = create_attention_mask(hidden_states, cache)

            if cache is None:
                cache = [None] * len(model.model.layers)

            for layer_index, (layer, c) in enumerate(zip(model.model.layers, cache)):
                attention_output = layer.self_attn(layer.input_layernorm(hidden_states), attention_mask, c)
                hidden_states = hidden_states + attention_output
                
                mlp_output = layer.mlp(layer.post_attention_layernorm(hidden_states))
                hidden_states = hidden_states + mlp_output
                if layer_index in control_layers:
                    hidden_states = apply_control_vector(hidden_states, layer_index)  # Apply control after MLP
            
            hidden_states = model.model.norm(hidden_states)

            # done calling model

            model_output = model.lm_head(hidden_states)
            logits = model_output[:, -1, :]

            y, probs = sample(logits)

            return y, probs, (hidden_states if return_hidden_states else None)

        y, p, h = _step(y)
        mx.async_eval(y)
        while True:
            next_y, next_p, next_h = _step(y)
            mx.async_eval(next_y)
            mx.eval(y)
            yield y, p, h
            y, p, h = next_y, next_p, next_h

    output_toks = []
    for step_output in generate_step_with_control(prompts_toks, model, **kwargs):
        tokens = step_output[0]  # Assuming tokens are the first element
        output_toks.append(tokens)
        if len(output_toks) >= max_tokens:
            break
        
        if len(output_toks) == 1:
            prompt_time = time.perf_counter() - start_time
            start_time = time.perf_counter()

    output_toks = mx.concatenate(output_toks, axis=1)

    responses = [
        response.split(tokenizer.eos_token)[0].split(tokenizer.pad_token)[0]
        for response in tokenizer.batch_decode(output_toks.tolist())
    ]
    
    if verbose:
        generation_time = time.perf_counter() - start_time
        prompt_tokens_per_second = prompts_toks.size / prompt_time
        generation_tokens_per_second = output_toks.size / generation_time
        print(f"Prompt: {prompt_tokens_per_second:.3f} tokens-per-second")
        print(f"Generation: {generation_tokens_per_second:.3f} tokens-per-second")
        for prompt, response in zip(prompts, responses):
            print("=" * 10)
            print("Prompt:", prompt)
            print(response)
            
    return responses