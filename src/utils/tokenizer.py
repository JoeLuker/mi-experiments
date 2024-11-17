from typing import List, Dict, Any, Union
from transformers import PreTrainedTokenizer
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class TokenizerWrapper:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self._tokenizer = tokenizer
        self.pad_token = tokenizer.pad_token
        self.eos_token = tokenizer.eos_token
        self.eos_token_id = tokenizer.eos_token_id
        
    def __call__(self, texts: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        """Tokenize text(s)"""
        return self._tokenizer(texts, **kwargs)
    
    def decode(self, token_ids: List[int], **kwargs) -> str:
        """Decode token IDs to text"""
        return self._tokenizer.decode(token_ids, **kwargs)
    
    def apply_chat_template(
        self, 
        messages: List[Dict[str, str]], 
        add_generation_prompt: bool = True,
        tokenize: bool = True
    ) -> Union[str, Dict[str, Any]]:
        """Apply chat template to messages"""
        formatted = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt
        )
        
        if tokenize:
            return self._tokenizer(formatted)
        return formatted
