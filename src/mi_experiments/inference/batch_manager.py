from typing import List, Dict, Optional, Generator
import mlx.core as mx
from dataclasses import dataclass
import time

from mi_experiments.core.cache import BatchedKVCache
from mi_experiments.utils.logging import setup_logger
from mi_experiments.inference.generate import generate_step

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
        self.cache = {}  # Store KV cache per batch
        
    def create_batches(
        self,
        sequences: List[str],
        return_attention: bool = False
    ) -> Generator[Dict[str, mx.array], None, None]:
        """Create optimized batches from input sequences."""
        # Tokenize all sequences
        tokenized = self.tokenizer._tokenizer(
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
    
    def get_batch_cache(self, batch_size: int) -> List[BatchedKVCache]:
        """Create or retrieve KV cache for batch."""
        cache_key = f"batch_{batch_size}"
        if cache_key not in self.cache:
            kv_heads = (
                [self.model.n_kv_heads] * len(self.model.layers)
                if isinstance(self.model.n_kv_heads, int)
                else self.model.n_kv_heads
            )
            
            self.cache[cache_key] = [
                BatchedKVCache(
                    head_dim=self.model.head_dim,
                    n_kv_heads=n_heads,
                    batch_size=batch_size
                )
                for n_heads in kv_heads
            ]
            
        return self.cache[cache_key]
    
    def process_batch(
        self,
        batch: Dict[str, mx.array],
        max_tokens: int = 100,
        **kwargs
    ) -> mx.array:
        """Process a single batch through the model for generation."""
        output_toks = []
        for step_output in generate_step(batch['input_ids'], self.model, **kwargs):
            tokens = step_output[0]  # Assuming tokens are the first element
            output_toks.append(tokens)
            if len(output_toks) >= max_tokens:
                break
        
        return mx.concatenate(output_toks, axis=1)

    def batch_generate(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        verbose: bool = False,
        format_prompts: bool = True,
        **kwargs,
    ) -> List[str]:
        """
        Generate responses for multiple prompts in batches.

        Args:
            prompts (List[str]): The list of string prompts.
            max_tokens (int): The maximum number of tokens. Default: ``100``.
            verbose (bool): If ``True``, print tokens and timing information.
                Default: ``False``.
            format_prompts (bool): If ``True``, format the prompts before tokenizing.
                Default: ``True``.
            kwargs: Additional options passed to generate_step.
        """
        if verbose:
            print("=" * 10)
        
        if format_prompts:
            prompts_fm = [[{"role": "user", "content": prompt}] for prompt in prompts]
            prompts_fm = [
                self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
                for prompt in prompts_fm
            ]
        else:
            prompts_fm = prompts

        # Left-padding for batched generation
        self.tokenizer._tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer._tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer._tokenizer.pad_token_id = self.tokenizer.eos_token_id

        tic = time.perf_counter()
        all_outputs = []
        
        for batch in self.create_batches(prompts_fm):
            output_toks = self.process_batch(batch, max_tokens=max_tokens, **kwargs)
            all_outputs.append(output_toks)
            
            if len(all_outputs) == 1:
                prompt_time = time.perf_counter() - tic
                tic = time.perf_counter()

        output_toks = mx.concatenate(all_outputs, axis=0)

        # Detokenizing + stripping pad/eos tokens
        responses = [
            response.split(self.tokenizer.eos_token)[0].split(self.tokenizer.pad_token)[0]
            for response in self.tokenizer.batch_decode(output_toks.tolist())
        ]
        
        if verbose:
            gen_time = time.perf_counter() - tic
            total_prompt_tokens = sum(len(self.tokenizer.encode(p)) for p in prompts_fm)
            total_output_tokens = output_toks.size
            prompt_tps = total_prompt_tokens / prompt_time
            gen_tps = total_output_tokens / gen_time
            print(f"Prompt: {prompt_tps:.3f} tokens-per-second")
            print(f"Generation: {gen_tps:.3f} tokens-per-second")
            for prompt, response in zip(prompts, responses):
                print("=" * 10)
                print("Prompt:", prompt)
                print(response)
                
        return responses


def batch_generate(
    model,
    tokenizer,
    prompts: List[str],
    max_tokens: int = 100,
    verbose: bool = False,
    format_prompts: bool = True,
    **kwargs,
) -> List[str]:
    """
    Standalone function to generate responses for multiple prompts in batches.
    """
    manager = BatchManager(model, tokenizer)
    return manager.batch_generate(
        prompts=prompts,
        max_tokens=max_tokens,
        verbose=verbose,
        format_prompts=format_prompts,
        **kwargs
    )