from typing import List, Dict, Optional, Generator, Tuple, Any
import mlx.core as mx
import mlx.nn as nn

from mi_experiments.core.cache import BatchedKVCache
from mi_experiments.utils.sampling import top_p_sampling

def apply_repetition_penalty(logits: mx.array, generated_tokens: Any, penalty: float):
    if len(generated_tokens) > 0:
        indices = mx.array([token for token in generated_tokens])
        selected_logits = logits[:, indices]
        selected_logits = mx.where(
            selected_logits < 0, selected_logits * penalty, selected_logits / penalty
        )
        logits[:, indices] = selected_logits
    return logits


def generate_step(
    prompts: mx.array,
    model: nn.Module,
    temp: float = 0.0,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = 20,
    top_p: float = 1.0,
    logit_bias: Optional[Dict[int, float]] = None,
    return_hidden_states: bool = False,
) -> Generator[Tuple[mx.array, mx.array, Optional[List[mx.array]]], None, None]:
    """
    A generator producing token ids based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        temp (float): The temperature for sampling, if 0 the argmax is used.
          Default: ``0``.
        repetition_penalty (float, optional): The penalty factor for repeating
          tokens.
        repetition_context_size (int, optional): The number of tokens to
          consider for repetition penalty. Default: ``20``.
        top_p (float, optional): Nulceus sampling, higher means model considers
          more less likely words.
        return_hidden_states (bool): If True, return hidden statesaqs. Default: ``False``.

    Yields:
        Generator[Tuple[mx.array, mx.array, Optional[List[mx.array]]]]: A generator producing
        one token, probability, and optionally hidden states per call.
    """

    def sample(logits: mx.array) -> Tuple[mx.array, float]:
        if logit_bias:
            indices = mx.array(list(logit_bias.keys()))
            values = mx.array(list(logit_bias.values()))
            logits[:, indices] += values
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

    if repetition_penalty:
        raise NotImplementedError("repetition_penalty not supported.")

    if repetition_penalty and (
        repetition_penalty < 0 or not isinstance(repetition_penalty, float)
    ):
        raise ValueError(
            f"repetition_penalty must be a non-negative float, got {repetition_penalty}"
        )

    # (bs, ntoks)
    y = prompts
    kv_heads = (
        [model.n_kv_heads] * len(model.layers)
        if isinstance(model.n_kv_heads, int)
        else model.n_kv_heads
    )

    cache = [BatchedKVCache(model.head_dim, n, y.shape[0]) for n in kv_heads]

    repetition_context = prompts

    if repetition_context_size and repetition_penalty:
        repetition_context = repetition_context[:,-repetition_context_size:]

    def _step(y):
        nonlocal repetition_context
        model_output = model(y, cache=cache, return_hidden_states=return_hidden_states)
        if return_hidden_states:
            logits, hidden_states = model_output
        else:
            logits = model_output
            hidden_states = None
        logits = logits[:, -1, :]

        if repetition_penalty:
            logits = apply_repetition_penalty(
                logits, repetition_context, repetition_penalty
            )
            y, probs = sample(logits)
            repetition_context = mx.concatenate([repetition_context, y])
        else:
            y, probs = sample(logits)

        if repetition_context_size:
            if repetition_context.shape[1] > repetition_context_size:
                repetition_context = repetition_context[:,-repetition_context_size:]


        return y, probs, hidden_states

    y, p, h = _step(y)
    mx.async_eval(y)
    while True:
        next_y, next_p, next_h = _step(y)
        mx.async_eval(next_y)
        mx.eval(y)
        yield y, p, h
        y, p, h = next_y, next_p, next_h
