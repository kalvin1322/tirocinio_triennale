"""
Model parameters configuration loader
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional

CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "model_parameters.json"


def load_model_parameters() -> Dict[str, Any]:
    """Load model parameters configuration"""
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)


def get_model_config(model_name: str) -> Optional[Dict[str, Any]]:
    """Get configuration for a specific model"""
    config = load_model_parameters()
    return config.get(model_name)


def get_default_params(model_name: str) -> Dict[str, Any]:
    """Get default parameters for a model"""
    model_config = get_model_config(model_name)
    if not model_config:
        return {}
    return model_config.get('default_params', {})


def get_tunable_params(model_name: str) -> Dict[str, Any]:
    """Get tunable parameters for a model"""
    model_config = get_model_config(model_name)
    if not model_config:
        return {}
    return model_config.get('tunable_params', {})


def build_model_params(model_name: str, **custom_params) -> Dict[str, Any]:
    """
    Build complete model parameters by merging defaults with custom values.
    
    Args:
        model_name: Name of the model
        **custom_params: Custom parameter values (will override defaults)
    
    Returns:
        Dictionary with all model parameters
    """
    # Start with default params
    params = get_default_params(model_name).copy()
    
    # Get tunable params info
    tunable = get_tunable_params(model_name)
    
    # Override with custom params (only if they're tunable or default)
    for key, value in custom_params.items():
        if value is not None:  # Only override if explicitly provided
            if key in tunable or key in params:
                params[key] = value
    
    return params


def validate_param(model_name: str, param_name: str, value: Any) -> tuple[bool, str]:
    """
    Validate a parameter value against model configuration.
    
    Returns:
        (is_valid, error_message)
    """
    tunable = get_tunable_params(model_name)
    
    if param_name not in tunable:
        return False, f"Parameter '{param_name}' is not tunable for {model_name}"
    
    param_config = tunable[param_name]
    param_type = param_config.get('type')
    
    # Type check
    if param_type == 'int' and not isinstance(value, int):
        return False, f"Parameter '{param_name}' must be an integer"
    
    # Range check
    if 'min' in param_config and value < param_config['min']:
        return False, f"Parameter '{param_name}' must be >= {param_config['min']}"
    
    if 'max' in param_config and value > param_config['max']:
        return False, f"Parameter '{param_name}' must be <= {param_config['max']}"
    
    # Options check
    if 'options' in param_config and value not in param_config['options']:
        return False, f"Parameter '{param_name}' must be one of {param_config['options']}"
    
    return True, ""


def load_checkpoint_metadata(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """
    Load metadata from a checkpoint's companion JSON file.
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
    
    Returns:
        Dictionary with metadata if JSON file exists, None otherwise
    
    Example:
        >>> metadata = load_checkpoint_metadata("models/FBP_UNet_V1_enc2_ch32.pth")
        >>> print(metadata['model_parameters'])
        {'num_encoders': 2, 'start_middle_channels': 32}
    """
    checkpoint_path = Path(checkpoint_path)
    metadata_path = checkpoint_path.with_suffix('.json')
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    return None


def get_model_filename(preprocessing: str, postprocessing: str, epochs: int = None, lr: float = None, **params) -> str:
    """
    Generate a simple, readable model filename.
    
    Since we now use JSON metadata for parameters, the filename just needs to be unique and human-readable.
    We include a timestamp to ensure uniqueness.
    
    Examples: 
        - FBP_UNet_V1_ep50_lr0001_20251107_143000.pth
        - SART_PostProcessNet_ep100_lr00001_20251107_150000.pth
    """
    from datetime import datetime
    
    # Base name with preprocessing and model
    base_name = f"{preprocessing}_{postprocessing}"
    
    # Add training parameters if provided
    training_parts = []
    if epochs is not None:
        training_parts.append(f"ep{epochs}")
    
    if lr is not None:
        # Format learning rate: 0.001 -> lr0001, 0.0001 -> lr00001
        lr_str = f"{lr:.6f}".replace('.', '').lstrip('0') or '0'
        training_parts.append(f"lr{lr_str}")
    
    if training_parts:
        base_name += "_" + "_".join(training_parts)
    
    # Add timestamp for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name += f"_{timestamp}"
    
    return f"{base_name}.pth"




