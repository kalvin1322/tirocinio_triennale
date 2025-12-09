"""
Refining methods registry for CT reconstruction.

This module provides a registry pattern to dynamically register and retrieve
refining methods.

To add a new refining method:
1. Create your reconstruction function in src/models/
2. Register it in this file using @register_preprocessing decorator or register_preprocessing()
3. Update configs/models_config.json with the method configuration

Example:
    >>> from utils.preprocessing_registry import register_preprocessing
    >>> 
    >>> @register_preprocessing("MY_METHOD")
    >>> def run_my_reconstruction(vol_geom, sinogram_id, **kwargs):
    ...     # Your reconstruction logic here
    ...     return reconstruction
"""



from typing import Callable, Dict, Any
import numpy as np

# Global registry to store preprocessing methods
_PREPROCESSING_REGISTRY: Dict[str, Callable] = {}


def register_refining(name: str):
    """
    Decorator to register a refining method.
    
    Args:
        name: Name of the refining method (e.g., "FISTA_TV", "CHAMBOLLE_POCK", "ADMM_TV")
    
    Example:
        >>> @register_refining("FISTA_TV")
        >>> def run_fista_tv_reconstruction(vol_geom, sinogram_id, **kwargs):
        ...     return reconstruction
    """
    def decorator(func: Callable):
        _PREPROCESSING_REGISTRY[name.upper()] = func
        return func
    return decorator

def get_refining_method(name: str) -> Callable:
    """
    Get a registered refining method by name.
    
    Args:
        name: Name of the refining method
    
    Returns:
        Callable: The refining function
    
    Raises:
        ValueError: If the refining method is not registered
    """
    name_upper = name.upper()
    if name_upper not in _PREPROCESSING_REGISTRY:
        raise ValueError(f"Refining method '{name}' is not registered.")
    return _PREPROCESSING_REGISTRY[name_upper]
def get_all_refining_methods() -> Dict[str, Callable]:
    """
    Get all registered refining methods.
    
    Returns:
        Dict[str, Callable]: Dictionary of all registered refining methods
    """
    return _PREPROCESSING_REGISTRY.copy()
from models.Fista_Tv_recostruction import run_fista_tv_reconstruction
@register_refining("FISTA_TV")
def fista_tv_wrapper(iterations=50, lambda_tv=0.1, **kwargs):
    """Wrapper per FISTA con Total Variation."""
    return run_fista_tv_reconstruction(
        iterations=iterations,
        lambda_tv=lambda_tv,
        **kwargs
    )

from models.chambolle_pock_reconstruction import run_chambolle_pock_reconstruction
@register_refining("CHAMBOLLE_POCK")
def chambolle_pock_wrapper(iterations=50, lambda_tv=0.1, **kwargs):
    """Wrapper per Chambolle-Pock."""
    return run_chambolle_pock_reconstruction(
        iterations=iterations,
        lambda_tv=lambda_tv,
        **kwargs
    )

from models.ADMM_tv_recostruction import run_admm_tv_reconstruction
@register_refining("ADMM_TV")
def admm_tv_wrapper(iterations=50, rho=1.0, lambda_tv=0.1, **kwargs):
    """Wrapper per ADMM con Total Variation."""
    return run_admm_tv_reconstruction(
        iterations=iterations,
        rho=rho,
        lambda_tv=lambda_tv,
        **kwargs
    )

