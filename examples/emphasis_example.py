import json
from pathlib import Path
from src.models import load_model
from src.inference.generator import self_sufficient_inference
from src.utils.logging import setup_logger

# Set up logging
logger = setup_logger(__name__, "emphasis_experiment.log")

def load_config(config_path: str = "configs/example_emphasis_config.json"):
    """Load experiment configuration from JSON file"""
    with open(config_path) as f:
        return json.load(f)

def run_emphasis_experiment(config_path: str = "configs/example_emphasis_config.json"):
    # Load configuration
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    
    # Load model
    logger.info("Loading model...")
    model, tokenizer = load_model(
        "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
    )

    # Get generation parameters
    gen_params = config["default_generation_params"]
    
    # Run experiments for each prompt
    for prompt in config["test_prompts"]:
        logger.info(f"\nTesting prompt: {prompt}")
        logger.info("=" * 80)
        
        # Run each experiment configuration
        for experiment in config["experiments"]:
            logger.info(f"\nRunning experiment: {experiment['name']}")
            logger.info(f"Description: {experiment['description']}")
            logger.info(f"Config: {experiment['config']}")
            
            response = self_sufficient_inference(
                model=model,
                tokenizer=tokenizer,
                prompts=[prompt],
                emphasis_config=experiment["config"],
                max_tokens=gen_params["max_tokens"],
                temp=gen_params["temperature"],
                top_p=gen_params["top_p"],
                verbose=True
            )
            
            logger.info(f"Response: {response[0]}\n")
            logger.info("-" * 80)

if __name__ == "__main__":
    # Allow custom config path via environment variable
    config_path = Path(
        os.getenv(
            "EMPHASIS_CONFIG_PATH", 
            "configs/example_emphasis_config.json"
        )
    )
    
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
        
    run_emphasis_experiment(str(config_path))