"""
Preprocessing methods registry for CT reconstruction.

This module provides a registry pattern to dynamically register and retrieve
preprocessing methods without modifying the CTDataloader code.

To add a new preprocessing method:
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


def register_preprocessing(name: str):
    """
    Decorator to register a preprocessing method.
    
    Args:
        name: Name of the preprocessing method (e.g., "FBP", "SART", "SIRT")
    
    Example:
        >>> @register_preprocessing("FBP")
        >>> def run_fbp_reconstruction(vol_geom, sinogram_id, **kwargs):
        ...     return reconstruction
    """
    def decorator(func: Callable):
        _PREPROCESSING_REGISTRY[name.upper()] = func
        return func
    return decorator


def get_preprocessing_method(name: str) -> Callable:
    """
    Get a registered preprocessing method by name.
    
    Args:
        name: Name of the preprocessing method
    
    Returns:
        Callable: The preprocessing function
    
    Raises:
        ValueError: If the preprocessing method is not registered
    """
    name_upper = name.upper()
    if name_upper not in _PREPROCESSING_REGISTRY:
        available = ", ".join(_PREPROCESSING_REGISTRY.keys())
        raise ValueError(
            f"Preprocessing method '{name}' not registered. "
            f"Available methods: {available}"
        )
    return _PREPROCESSING_REGISTRY[name_upper]


def list_preprocessing_methods() -> list[str]:
    """
    List all registered preprocessing methods.
    
    Returns:
        list[str]: List of registered method names
    """
    return list(_PREPROCESSING_REGISTRY.keys())


def is_registered(name: str) -> bool:
    """
    Check if a preprocessing method is registered.
    
    Args:
        name: Name of the preprocessing method
    
    Returns:
        bool: True if registered, False otherwise
    """
    return name.upper() in _PREPROCESSING_REGISTRY


# ============================================================================
# Register built-in preprocessing methods
# ============================================================================

from models.FBP_recostruction import run_fbp_reconstruction
from models.SART_recostruction import run_sart_reconstruction


@register_preprocessing("FBP")
def fbp_wrapper(vol_geom, sinogram_id, filter_type="ram-lak", use_cuda=True, **kwargs):
    """FBP (Filtered Back Projection) reconstruction wrapper."""
    return run_fbp_reconstruction(
        vol_geom=vol_geom,
        sinogram_id=sinogram_id,
        filter_type=filter_type,
        use_cuda=use_cuda
    )


@register_preprocessing("SART")
def sart_wrapper(vol_geom, sinogram_id, iterations=50, projector_type='linear', 
                 projection_order='sequential', relaxation_factor=1.0, **kwargs):
    """SART (Simultaneous Algebraic Reconstruction Technique) wrapper."""
    return run_sart_reconstruction(
        vol_geom=vol_geom,
        sinogram_id=sinogram_id,
        iterations=iterations,
        projector_type=projector_type,
        projection_order=projection_order,
        relaxation_factor=relaxation_factor
    )

from models.SIRT_recostruction import run_sirt_reconstruction
@register_preprocessing("SIRT")
def sirt_wrapper(vol_geom, sinogram_id, iterations=50, projector_type='linear', min_constraint=0.0, max_constraint=None):
    """SIRT (Simultaneous Iterative Reconstruction Technique) wrapper."""
    return run_sirt_reconstruction(
        vol_geom=vol_geom,
        sinogram_id=sinogram_id,
        iterations=iterations,
        projector_type=projector_type,
        min_constraint=min_constraint,
        max_constraint=max_constraint
    )

from models.Fista_Tv_recostruction import run_fista_tv_reconstruction
@register_preprocessing("FISTA_TV")
def fista_tv_wrapper(vol_geom, sinogram_id, iterations=50, lambda_tv=0.1, **kwargs):
    """Wrapper per FISTA con Total Variation."""
    # Nota: richiede che tu abbia creato il file src/models/Advanced_reconstruction.py
    return run_fista_tv_reconstruction(
        vol_geom=vol_geom,
        sinogram_id=sinogram_id,
        iterations=iterations,
        lambda_tv=lambda_tv
    )
from models.ADMM_tv_recostruction import run_admm_tv_reconstruction
@register_preprocessing("ADMM_TV")
def admm_tv_wrapper(vol_geom, sinogram_id, iterations=50, rho=1.0, lambda_tv=0.1, **kwargs):
    """Wrapper per ADMM con Total Variation."""
    return run_admm_tv_reconstruction(
        vol_geom=vol_geom,
        sinogram_id=sinogram_id,
        iterations=iterations,
        rho=rho,
        lambda_tv=lambda_tv
    )
# ============================================================================
# Example: How to add a new method (commented out)
# ============================================================================
"""
# Step 1: Import your reconstruction function
from models.SIRT_reconstruction import run_sirt_reconstruction

# Step 2: Register it
@register_preprocessing("SIRT")
def sirt_wrapper(vol_geom, sinogram_id, iterations=100, **kwargs):
    return run_sirt_reconstruction(
        vol_geom=vol_geom,
        sinogram_id=sinogram_id,
        iterations=iterations
    )

# That's it! The method is now available throughout the system.
"""
