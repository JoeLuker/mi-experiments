import pytest
import mlx.core as mx
from src.models import load_model
from src.inference.generator import generate_with_emphasis
from src.utils.config import EmphasisConfig, ConfigManager

@pytest.fixture
def test_setup():
    model, tokenizer = load_model("mlx-community/Mistral-7B-Instruct-v0.3-4bit")
    config_manager = ConfigManager()
    return model, tokenizer, config_manager

def test_end_to_end_inference(test_setup):
    model, tokenizer, _ = test_setup
    
    # Test basic inference
    prompts = ["Explain what a neural network is."]
    response = generate_with_emphasis(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        config={"max_tokens": 50, "temperature": 0.7, "top_p": 0.9}
    )
    assert isinstance(response[0], str)
    assert len(response[0]) > 0

def test_emphasis_configurations(test_setup):
    model, tokenizer, config_manager = test_setup
    
    # Reference example configurations
    emphasis_configs = []
    
    # Reference from emphasis_example.py
    for config in emphasis_configs:
        response = generate_with_emphasis(
            model=model,
            tokenizer=tokenizer,
            prompts=["Test prompt"],
            config={"max_tokens": 20},
            emphasis_config=config
        )
        assert isinstance(response[0], str)

def test_config_persistence(test_setup):
    _, _, config_manager = test_setup
    
    test_config = EmphasisConfig(
        layers={'0': 1.5, '1': 0.0},
        heads={'3': {'1': 2.0, '2': 0.0}},
        neurons={'4': {'15': 1.5, '30': 0.0}}
    )
    
    config_manager.emphasis_config = test_config
    config_manager.save_config()
    
    new_config_manager = ConfigManager()
    new_config_manager.load_config()
    
    assert new_config_manager.emphasis_config.to_dict() == test_config.to_dict()