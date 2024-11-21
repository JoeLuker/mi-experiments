from typing import Tuple
import mlx.core as mx
import mlx.nn as nn

from mlx_lm.tokenizer_utils import TokenizerWrapper, load_tokenizer

import logging
import json
import glob
from pathlib import Path
from typing import Optional


from huggingface_hub import snapshot_download
from huggingface_hub.errors import RepositoryNotFoundError

from mi_experiments.core.config import ModelArgs
from mi_experiments.core.model import Model


def load_config(model_path: Path) -> dict:
    try:
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        logging.error(f"Config file not found in {model_path}")
        raise
    return config

def load_model(
    model_path: Path,
    lazy: bool = False,
    model_config: dict = {},
) -> nn.Module:
    config = load_config(model_path)
    config.update(model_config)

    weight_files = glob.glob(str(model_path / "model*.safetensors"))
    if not weight_files:
        weight_files = glob.glob(str(model_path / "weight*.safetensors"))
    
    if not weight_files:
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    model_args = ModelArgs.from_dict(config)
    model = Model(model_args)

    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

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

    model.load_weights(list(weights.items()), strict=False)

    if not lazy:
        mx.eval(model.parameters())

    model.eval()
    return model


class ModelNotFoundError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

def get_model_path(path_or_hf_repo: str, revision: Optional[str] = None) -> Path:
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        try:
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
        except RepositoryNotFoundError:
            raise ModelNotFoundError(
                f"Model not found for path or HF repo: {path_or_hf_repo}.\n"
                "Please make sure you specified the local path or Hugging Face"
                " repo id correctly.\nIf you are trying to access a private or"
                " gated Hugging Face repo, make sure you are authenticated:\n"
                "https://huggingface.co/docs/huggingface_hub/en/guides/cli#huggingface-cli-login"
            ) from None
    return model_path

def load(
    path_or_hf_repo: str,
    tokenizer_config={},
    model_config={},
    lazy: bool = False,
) -> Tuple[nn.Module, TokenizerWrapper]:

    model_path = get_model_path(path_or_hf_repo)


    model = load_model(model_path, lazy, model_config)
    tokenizer = load_tokenizer(model_path, tokenizer_config)

    return model, tokenizer