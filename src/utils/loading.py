# utils/loading.py

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import glob

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx_lm.tokenizer_utils import TokenizerWrapper, load_tokenizer

from src.core.model import Model, ModelArgs

logger = logging.getLogger(__name__)

def load_config(model_path: Path) -> Dict:
    """Load model configuration from json file."""
    try:
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)
        logger.debug(f"Loaded config from {model_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found in {model_path}")
        raise

def get_model_path(path_or_hf_repo: str, revision: Optional[str] = None) -> Path:
    """Get local path for model, downloading from HF if needed."""
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        logger.info(f"Downloading model from {path_or_hf_repo}")
        model_path = Path(
            snapshot_download(
                repo_id=path_or_hf_repo,
                revision=revision,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.py",
                    "tokenizer.model",
                    "*.tiktoken",
                    "*.txt",
                ],
            )
        )
    return model_path

def load_model(
    model_path: Path,
    lazy: bool = False,
    model_config: Dict = {},
) -> nn.Module:
    """Load model from path with optional configuration."""
    config = load_config(model_path)
    config.update(model_config)

    # Find model weights
    weight_files = glob.glob(str(model_path / "model*.safetensors"))
    if not weight_files:
        weight_files = glob.glob(str(model_path / "weight*.safetensors"))
    
    if not weight_files:
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    # Load weights
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    # Initialize model
    model_args = ModelArgs.from_dict(config)
    model = Model(model_args)

    # Handle quantization
    if (quantization := config.get("quantization", None)) is not None:
        def class_predicate(p, m):
            if not hasattr(m, "to_quantized"):
                return False
            return f"{p}.scales" in weights

        nn.quantize(
            model,
            **quantization,
            class_predicate=class_predicate,
        )

    # Load weights
    model.load_weights(list(weights.items()), strict=False)

    if not lazy:
        mx.eval(model.parameters())

    model.eval()
    return model

def load(
    path_or_hf_repo: str,
    tokenizer_config: Dict = {},
    model_config: Dict = {},
    lazy: bool = False,
) -> Tuple[nn.Module, TokenizerWrapper]:
    """Load both model and tokenizer."""
    model_path = get_model_path(path_or_hf_repo)
    
    model = load_model(model_path, lazy, model_config)
    tokenizer = load_tokenizer(model_path, tokenizer_config)

    return model, tokenizer