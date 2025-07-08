"""Configuration module for document layout detection."""
import yaml
from pathlib import Path
from typing import Dict, Any

# Path to the main config file
CONFIG_PATH = Path(__file__).parent / 'training_config.yaml'

def load_config() -> Dict[str, Any]:
    """Load and return the configuration dictionary.
    
    Returns:
        Dict containing the configuration parameters
    """
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def get_class_names() -> Dict[int, str]:
    """Get the class names mapping from the config.
    
    Returns:
        Dict mapping class IDs to class names
    """
    config = load_config()
    return config['model']['class_names']

def get_num_classes() -> int:
    """Get the number of classes from the config.
    
    Returns:
        Number of classes
    """
    config = load_config()
    return config['model']['num_classes']
