"""
SART (Simultaneous Algebraic Reconstruction Technique) reconstruction implementation.

This module provides a functional interface for SART-based CT reconstruction using ASTRA toolbox.
"""

import astra
import numpy as np
from typing import Optional, Dict, Any


def sart_reconstruction(
    sinogram: np.ndarray,
    geometry_config: Dict[str, Any],
    iterations: int = 100,
    projector_type: str = 'linear',
    projection_order: Optional[str] = 'sequential',
    relaxation_factor: float = 1.0
) -> np.ndarray:
    """
    Perform SART (Simultaneous Algebraic Reconstruction Technique) reconstruction.
    
    SART is an iterative algebraic reconstruction technique that updates the reconstruction
    by processing projections sequentially or in a custom order.
    
    Args:
        sinogram (np.ndarray): Input sinogram data (projections)
        geometry_config (Dict[str, Any]): Geometry configuration containing:
            - image_size: Size of the reconstruction image (N x N)
            - num_angles: Number of projection angles
            - det_spacing: Detector spacing
            - angles: Array of projection angles (optional, defaults to uniform [0, π])
        iterations (int): Number of SART iterations to perform (default: 100)
        projector_type (str): Type of projector ('linear', 'cuda', etc.) (default: 'linear')
        projection_order (str): Order of projection updates:
            - 'sequential': Process projections in order 0, 1, 2, ...
            - 'random': Random order
            - 'custom': Custom interleaved pattern
            (default: 'sequential')
        relaxation_factor (float): Relaxation parameter (0 < λ ≤ 1) (default: 1.0)
    
    Returns:
        np.ndarray: Reconstructed image (N x N)
    
    Example:
        >>> sinogram = np.random.rand(180, 256)
        >>> geometry = {
        ...     'image_size': 256,
        ...     'num_angles': 180,
        ...     'det_spacing': 1.0
        ... }
        >>> reconstruction = sart_reconstruction(sinogram, geometry, iterations=100)
    """
    # Extract geometry parameters
    image_size = geometry_config.get('image_size', 256)
    num_angles = geometry_config.get('num_angles', 180)
    det_spacing = geometry_config.get('det_spacing', 1.0)
    angles = geometry_config.get('angles', np.linspace(0, np.pi, num_angles, endpoint=False))
    
    # Create ASTRA geometries
    proj_geom = astra.create_proj_geom('parallel', det_spacing, sinogram.shape[1], angles)
    vol_geom = astra.create_vol_geom(image_size, image_size)
    
    # Create projector
    projector_id = astra.create_projector(projector_type, proj_geom, vol_geom)
    
    # Create ASTRA data objects
    sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram)
    recon_id = astra.data2d.create('-vol', vol_geom)
    
    # Configure SART algorithm
    cfg = astra.astra_dict('SART')
    cfg['ProjectorId'] = projector_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['ReconstructionDataId'] = recon_id
    
    # Set projection order
    if projection_order == 'custom':
        # Custom interleaved pattern: 0, 5, 10, ..., 1, 6, 11, ..., 2, 7, ...
        stride = 5
        projection_order_list = []
        for offset in range(stride):
            projection_order_list.extend(range(offset, num_angles, stride))
        cfg['option'] = {'ProjectionOrder': 'custom'}
        cfg['option']['ProjectionOrderList'] = np.array(projection_order_list, dtype=np.int32)
    elif projection_order == 'random':
        cfg['option'] = {'ProjectionOrder': 'random'}
    # else: sequential is the default
    
    # Set relaxation factor if different from default
    if relaxation_factor != 1.0:
        if 'option' not in cfg:
            cfg['option'] = {}
        cfg['option']['Relaxation'] = relaxation_factor
    
    # Create and run algorithm
    algorithm_id = astra.algorithm.create(cfg)
    astra.algorithm.run(algorithm_id, iterations)
    
    # Get reconstruction result
    reconstruction = astra.data2d.get(recon_id)
    
    # Clean up ASTRA objects
    astra.data2d.delete([sinogram_id, recon_id])
    astra.projector.delete(projector_id)
    astra.algorithm.delete(algorithm_id)
    
    return reconstruction


def run_sart_reconstruction(
    vol_geom,
    sinogram_id: int,
    iterations: int = 50,
    projector_type: str = 'linear',
    projection_order: str = 'sequential',
    relaxation_factor: float = 1.0
) -> np.ndarray:
    """
    Performs a single SART reconstruction using ASTRA (compatible with CTDataset).
    
    This function is a simplified wrapper that works with ASTRA data structures directly,
    similar to run_fbp_reconstruction.
    
    Args:
        vol_geom: The ASTRA volume geometry (created with astra.create_vol_geom)
        sinogram_id: The ASTRA ID of the sinogram data (created with astra.data2d.create)
        iterations: Number of SART iterations to perform (default: 50)
        projector_type: Type of projector ('linear', 'cuda', etc.) (default: 'linear')
        projection_order: Order of projection updates ('sequential', 'random', 'custom') (default: 'sequential')
        relaxation_factor: Relaxation parameter (0 < λ ≤ 1) (default: 1.0)
    
    Returns:
        np.ndarray: The reconstructed image as a NumPy array
    """
    # Get projection geometry from sinogram
    proj_geom = astra.data2d.get_geometry(sinogram_id)
    num_angles = proj_geom['ProjectionAngles'].shape[0]
    
    # Create projector
    projector_id = astra.create_projector(projector_type, proj_geom, vol_geom)
    
    # Create reconstruction volume
    recon_id = astra.data2d.create('-vol', vol_geom, data=0.0)
    
    # Configure SART algorithm
    cfg = astra.astra_dict('SART')
    cfg['ProjectorId'] = projector_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['ReconstructionDataId'] = recon_id
    
    # Set projection order
    if projection_order == 'custom':
        # Custom interleaved pattern: 0, 5, 10, ..., 1, 6, 11, ..., 2, 7, ...
        stride = 5
        projection_order_list = []
        for offset in range(stride):
            projection_order_list.extend(range(offset, num_angles, stride))
        cfg['option'] = {'ProjectionOrder': 'custom'}
        cfg['option']['ProjectionOrderList'] = np.array(projection_order_list, dtype=np.int32)
    elif projection_order == 'random':
        cfg['option'] = {'ProjectionOrder': 'random'}
    # else: sequential is the default
    
    # Set relaxation factor if different from default
    if relaxation_factor != 1.0:
        if 'option' not in cfg:
            cfg['option'] = {}
        cfg['option']['Relaxation'] = relaxation_factor
    
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