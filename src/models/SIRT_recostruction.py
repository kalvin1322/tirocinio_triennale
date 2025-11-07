"""
SIRT (Simultaneous Iterative Reconstruction Technique) reconstruction implementation.

This module provides a functional interface for SIRT-based CT reconstruction 
using the ASTRA toolbox.
"""

import astra
import numpy as np
from typing import Optional, Dict, Any


def run_sirt_reconstruction(
    vol_geom,
    sinogram_id: int,
    iterations: int = 50,
    projector_type: str = 'linear',
    min_constraint: Optional[float] = 0.0,
    max_constraint: Optional[float] = None
) -> np.ndarray:
    """
    Performs a single SIRT reconstruction using ASTRA (compatible with CTDataset).
    
    This function is a simplified wrapper that works with ASTRA data structures directly.
    
    Args:
        vol_geom: The ASTRA volume geometry (created with astra.create_vol_geom).
        sinogram_id: The ASTRA ID of the sinogram data (created with astra.data2d.create).
        iterations (int): Number of SIRT iterations to perform (default: 50).
        projector_type (str): Type of projector ('linear', 'cuda', etc.) (default: 'linear').
        min_constraint (Optional[float]): Minimum value constraint (default: 0.0).
        max_constraint (Optional[float]): Maximum value constraint (default: None).
    
    Returns:
        np.ndarray: The reconstructed image as a NumPy array.
    """
    # Get projection geometry from sinogram
    proj_geom = astra.data2d.get_geometry(sinogram_id)
    
    # Create projector
    projector_id = astra.create_projector(projector_type, proj_geom, vol_geom)
    
    # Create reconstruction volume
    recon_id = astra.data2d.create('-vol', vol_geom, data=0.0)
    
    # Configure SIRT algorithm
    cfg = astra.astra_dict('SIRT')
    cfg['ProjectorId'] = projector_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['ReconstructionDataId'] = recon_id
    
    # Set constraints
    options = {}
    if min_constraint is not None:
        options['MinConstraint'] = min_constraint
    if max_constraint is not None:
        options['MaxConstraint'] = max_constraint
    if options:
        cfg['option'] = options
        
    # Create and run algorithm
    algorithm_id = astra.algorithm.create(cfg)
    
    try:
        astra.algorithm.run(algorithm_id, iterations)
        reconstruction = astra.data2d.get(recon_id)
    finally:
        # Clean up ASTRA objects
        astra.data2d.delete(recon_id)
        astra.projector.delete(projector_id)
        astra.algorithm.delete(algorithm_id)
    
    return reconstruction