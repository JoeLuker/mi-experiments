from pathlib import Path
import os

def get_model_path(model_name: str) -> Path:
    """Resolve model path with fallbacks"""
    # Check environment variable first
    if model_dir := os.getenv("MODEL_DIR"):
        path = Path(model_dir) / model_name
        if path.exists():
            return path
            
    # Check common locations
    locations = [
        Path.home() / ".cache/huggingface/hub",
        Path("/models"),
        Path.cwd() / "models"
    ]
    
    for loc in locations:
        path = loc / model_name
        if path.exists():
            return path
            
    raise FileNotFoundError(f"Could not find model: {model_name}") 