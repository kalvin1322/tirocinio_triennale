"""
Postprocessing models registry for CT reconstruction.

This module provides a registry pattern to dynamically register and retrieve
postprocessing models without modifying the command code.

To add a new postprocessing model:
1. Create your model class in src/models/
2. Register it in this file using @register_postprocessing decorator
3. Update configs/models_config.json with the model configuration

Example:
    >>> from utils.postprocessing_registry import register_postprocessing
    >>> from models.MyModel import MyCustomModel
    >>> 
    >>> @register_postprocessing("MyModel")
    >>> def my_model_factory(**kwargs):
    ...     return MyCustomModel(**kwargs)
"""

from typing import Callable, Dict, Any
import torch.nn as nn

# Global registry to store postprocessing model factories
_POSTPROCESSING_REGISTRY: Dict[str, Callable] = {}


def register_postprocessing(name: str):
    """
    Decorator to register a postprocessing model factory.
    
    Args:
        name: Name of the postprocessing model (e.g., "UNet_V1", "ThreeL_SSNet")
    
    Example:
        >>> @register_postprocessing("UNet_V1")
        >>> def unet_factory(in_channels=1, out_channels=1, **kwargs):
        ...     return UNet_V1(in_channels, out_channels, **kwargs)
    """
    def decorator(func: Callable):
        _POSTPROCESSING_REGISTRY[name] = func
        return func
    return decorator


def get_postprocessing_model(name: str, **kwargs) -> nn.Module:
    """
    Get a registered postprocessing model instance by name.
    
    Args:
        name: Name of the postprocessing model
        **kwargs: Parameters to pass to the model constructor
    
    Returns:
        nn.Module: The instantiated model
    
    Raises:
        ValueError: If the postprocessing model is not registered
    """
    if name not in _POSTPROCESSING_REGISTRY:
        available = ", ".join(_POSTPROCESSING_REGISTRY.keys())
        raise ValueError(
            f"Postprocessing model '{name}' not registered. "
            f"Available models: {available}"
        )
    return _POSTPROCESSING_REGISTRY[name](**kwargs)


def list_postprocessing_models() -> list[str]:
    """
    List all registered postprocessing models.
    
    Returns:
        list[str]: List of registered model names
    """
    return list(_POSTPROCESSING_REGISTRY.keys())


def is_registered(name: str) -> bool:
    """
    Check if a postprocessing model is registered.
    
    Args:
        name: Name of the postprocessing model
    
    Returns:
        bool: True if registered, False otherwise
    """
    return name in _POSTPROCESSING_REGISTRY


# ============================================================================
# Register built-in postprocessing models
# ============================================================================

from models.UNet_V1 import UNet_V1
from models.ThreeL_SSNet import ThreeL_SSNet
from models.SimpleResNet import SimpleResNet
from models.PostProcessNet import PostProcessNet

@register_postprocessing("UNet_V1")
def unet_v1_factory(in_channels=1, out_channels=1, num_encoders=3, start_middle_channels=32, **kwargs):
    """UNet V1 model factory with configurable architecture parameters."""
    return UNet_V1(
        in_channels=in_channels, 
        out_channels=out_channels,
        num_encoders=num_encoders,
        start_middle_channels=start_middle_channels,
        **kwargs
    )


@register_postprocessing("ThreeL_SSNet")
def threelssnet_factory(**kwargs):
    """
    Three-Level Squeeze-and-Excitation Network factory.
    Note: This model has no configurable parameters - architecture is fixed.
    """
    return ThreeL_SSNet()

@register_postprocessing("SimpleResNet")
def simpleresnet_factory(in_channels=1, out_channels=1, num_layers=4, features=32, **kwargs):
    """Simple Residual Network factory with configurable parameters."""
    return SimpleResNet(
        in_channels=in_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        features=features,
        **kwargs
    )
@register_postprocessing("PostProcessNet")
def postprocessnet_factory(in_channels=1, out_channels=1, hidden_channels=32, use_residual=True, **kwargs):
    """Post Processing Network factory with configurable parameters."""
    return PostProcessNet(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        use_residual=use_residual,
        **kwargs
)
# ============================================================================
# Example: How to add a new model
# ============================================================================
"""
# Step 1: Import your model class
from models.ResNet_CT import ResNet_CT

# Step 2: Register it
@register_postprocessing("ResNet_CT")
def resnet_ct_factory(num_blocks=3, channels=64, **kwargs):
    return ResNet_CT(num_blocks=num_blocks, channels=channels, **kwargs)

# That's it! The model is now available throughout the system.
"""
