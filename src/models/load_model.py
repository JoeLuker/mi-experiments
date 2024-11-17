from typing import Tuple, Optional
import mlx.core as mx
from transformers import AutoTokenizer
from .transformer import ModelArgs, TransformerModel
from ..utils.logging import setup_logger
from ..utils.tokenizer import TokenizerWrapper

logger = setup_logger(__name__)

def load_model(
    model_name: str,
    quantize: bool = True,
    device: Optional[str] = None
) -> Tuple[TransformerModel, TokenizerWrapper]:
    """
    Load a pre-trained model and tokenizer
    """
    logger.info(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = TokenizerWrapper(tokenizer)
    
    # Load model config and create args
    config = AutoConfig.from_pretrained(model_name)
    model_args = ModelArgs(
        model_type=config.model_type,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        intermediate_size=config.intermediate_size,
        num_attention_heads=config.num_attention_heads,
        rms_norm_eps=config.rms_norm_eps,
        vocab_size=config.vocab_size,
        head_dim=getattr(config, "head_dim", None),
        num_key_value_heads=getattr(config, "num_key_value_heads", None),
        attention_bias=getattr(config, "attention_bias", False),
        mlp_bias=getattr(config, "mlp_bias", False)
    )
    
    # Initialize model
    model = TransformerModel(model_args)
    
    # Load weights
    weights = mx.load(f"{model_name}/weights.safetensors")
    if quantize:
        weights = {k: quantize_weights(v) for k, v in weights.items()}
    model.update(weights)
    
    logger.info("Model loaded successfully")
    return model, tokenizer 