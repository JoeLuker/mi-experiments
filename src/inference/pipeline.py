# inference/pipeline.py

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union, Generator, Callable
from threading import Lock

import mlx.core as mx
from transformers import PreTrainedTokenizer

from .generate import GenerationConfig, generate
from ..core.model import Model
from ..utils.sampling import sample_token

logger = logging.getLogger(__name__)

@dataclass
class GenerationPipeline:
    """High-level pipeline for text generation."""
    
    model: Model
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper]
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    
    # Optional configurations
    emphasis_config: Optional[Dict[str, Any]] = None
    stop_words: List[str] = field(default_factory=list)
    max_new_tokens: int = 100
    stream: bool = False
    
    # Thread safety for batch processing
    _lock: Lock = field(default_factory=Lock)

    def __post_init__(self):
        """Initialize pipeline specific settings."""
        self.model.eval()
        if not isinstance(self.tokenizer, TokenizerWrapper):
            self.tokenizer = TokenizerWrapper(self.tokenizer)

    def _prepare_prompts(
        self,
        prompts: Union[str, List[str]],
        system_prompt: Optional[str] = None,
    ) -> List[str]:
        """Prepare prompts for generation."""
        if isinstance(prompts, str):
            prompts = [prompts]
            
        if system_prompt:
            prompts = [
                f"{system_prompt}\n\n{prompt}" for prompt in prompts
            ]
            
        return prompts

    def _process_stop_words(self, text: str) -> str:
        """Truncate text at stop words."""
        for stop_word in self.stop_words:
            if stop_word in text:
                text = text[:text.index(stop_word)]
        return text

    def _create_generation_callback(
        self,
        callback: Optional[Callable] = None
    ) -> Callable:
        """Create callback for generation monitoring."""
        def default_callback(step: int, token: str, output: str):
            if self.stream:
                print(token, end="", flush=True)
                
        return callback or default_callback

    def generate(
        self,
        prompts: Union[str, List[str]],
        system_prompt: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
        emphasis_config: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable] = None,
        **kwargs
    ) -> Union[List[str], Generator[List[str], None, None]]:
        """Generate text from prompts."""
        # Thread safety for batch processing
        with self._lock:
            # Prepare inputs
            prompts = self._prepare_prompts(prompts, system_prompt)
            generation_config = generation_config or self.generation_config
            emphasis_config = emphasis_config or self.emphasis_config
            
            # Update config with kwargs
            for k, v in kwargs.items():
                if hasattr(generation_config, k):
                    setattr(generation_config, k, v)

            # Create callback
            callback = self._create_generation_callback(callback)
            
            # Initialize generation
            responses: List[str] = [""] * len(prompts)
            
            try:
                # Generate text
                for step, (partial_responses, output) in enumerate(generate(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompts=prompts,
                    generation_config=generation_config,
                    emphasis_config=emphasis_config,
                )):
                    # Process responses
                    for i, (prev_response, new_response) in enumerate(zip(responses, partial_responses)):
                        # Get new token text
                        token_text = new_response[len(prev_response):]
                        
                        # Update response
                        responses[i] = new_response
                        
                        # Process stop words
                        if self.stop_words:
                            responses[i] = self._process_stop_words(responses[i])
                            
                        # Call callback
                        callback(step, token_text, responses[i])
                        
                    # Stream output if requested
                    if self.stream:
                        yield responses
                        
                    # Check for early stopping
                    if all(any(stop in resp for stop in self.stop_words) for resp in responses):
                        break
                        
                    # Check max tokens
                    if step >= self.max_new_tokens:
                        break

            except Exception as e:
                logger.error(f"Error during generation: {str(e)}")
                raise

            # Return final responses for non-streaming mode
            if not self.stream:
                return responses

    def batch_generate(
        self,
        prompts: List[str],
        batch_size: int = 4,
        **kwargs
    ) -> List[str]:
        """Generate text for multiple prompts in batches."""
        all_responses = []
        
        # Process prompts in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            responses = self.generate(batch_prompts, **kwargs)
            
            if isinstance(responses, Generator):
                # For streaming, get final responses
                for r in responses:
                    pass
                responses = r
                
            all_responses.extend(responses)
            
        return all_responses

    def __call__(
        self,
        prompts: Union[str, List[str]],
        **kwargs
    ) -> Union[List[str], Generator[List[str], None, None]]:
        """Convenient call method."""
        return self.generate(prompts, **kwargs)