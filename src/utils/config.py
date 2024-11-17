from dataclasses import dataclass
from typing import Dict, Any, Optional
import json
from pathlib import Path
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

@dataclass
class EmphasisConfig:
    layers: Optional[Dict[str, float]] = None
    heads: Optional[Dict[str, Dict[str, float]]] = None
    neurons: Optional[Dict[str, Dict[str, float]]] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EmphasisConfig':
        return cls(
            layers=config_dict.get('layers'),
            heads=config_dict.get('heads'),
            neurons=config_dict.get('neurons')
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'layers': self.layers,
            'heads': self.heads,
            'neurons': self.neurons
        }

class ConfigManager:
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path('configs/default_config.json')
        self.emphasis_config = EmphasisConfig()
        
    def load_config(self) -> None:
        """Load configuration from file"""
        try:
            with open(self.config_path) as f:
                config_dict = json.load(f)
            self.emphasis_config = EmphasisConfig.from_dict(config_dict)
            logger.info(f"Loaded config from {self.config_path}")
        except FileNotFoundError:
            logger.warning(f"Config file not found at {self.config_path}, using defaults")
            
    def save_config(self) -> None:
        """Save current configuration to file"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.emphasis_config.to_dict(), f, indent=2)
        logger.info(f"Saved config to {self.config_path}") 