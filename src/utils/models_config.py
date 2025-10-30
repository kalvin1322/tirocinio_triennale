"""
Models configuration loader
Manages preprocessing and postprocessing model configurations
"""
import json
from pathlib import Path
from typing import Dict, List, Optional


def load_models_config(config_path: str = "configs/models_config.json") -> Dict:
    """
    Load models configuration from JSON file
    
    Args:
        config_path: Path to models config file
        
    Returns:
        Dictionary with preprocessing and postprocessing models
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Models config not found at: {config_path}\n"
            "Please create the models configuration file."
        )
    
    with open(config_path, 'r') as f:
        return json.load(f)


def get_preprocessing_models(config_path: str = "configs/models_config.json") -> List[str]:
    """Get list of available preprocessing models"""
    config = load_models_config(config_path)
    return list(config.get('preprocessing', {}).keys())


def get_postprocessing_models(config_path: str = "configs/models_config.json") -> List[str]:
    """Get list of available postprocessing models"""
    config = load_models_config(config_path)
    return list(config.get('postprocessing', {}).keys())


def get_preprocessing_info(model_name: str, config_path: str = "configs/models_config.json") -> Dict:
    """Get information about a specific preprocessing model"""
    config = load_models_config(config_path)
    preprocessing = config.get('preprocessing', {})
    
    if model_name not in preprocessing:
        available = ", ".join(preprocessing.keys())
        raise ValueError(
            f"Preprocessing model '{model_name}' not found.\n"
            f"Available models: {available}"
        )
    
    return preprocessing[model_name]


def get_postprocessing_info(model_name: str, config_path: str = "configs/models_config.json") -> Dict:
    """Get information about a specific postprocessing model"""
    config = load_models_config(config_path)
    postprocessing = config.get('postprocessing', {})
    
    if model_name not in postprocessing:
        available = ", ".join(postprocessing.keys())
        raise ValueError(
            f"Postprocessing model '{model_name}' not found.\n"
            f"Available models: {available}"
        )
    
    return postprocessing[model_name]


def get_model_combinations(preprocessing_models: Optional[List[str]] = None,
                          postprocessing_models: Optional[List[str]] = None,
                          config_path: str = "configs/models_config.json") -> List[tuple]:
    """
    Generate all combinations of preprocessing and postprocessing models
    
    Args:
        preprocessing_models: List of preprocessing model names (None = all)
        postprocessing_models: List of postprocessing model names (None = all)
        config_path: Path to config file
        
    Returns:
        List of tuples (preprocessing_name, postprocessing_name)
    """
    config = load_models_config(config_path)
    
    if preprocessing_models is None:
        preprocessing_models = list(config.get('preprocessing', {}).keys())
    
    if postprocessing_models is None:
        postprocessing_models = list(config.get('postprocessing', {}).keys())
    
    combinations = []
    for prep in preprocessing_models:
        for postp in postprocessing_models:
            combinations.append((prep, postp))
    
    return combinations
