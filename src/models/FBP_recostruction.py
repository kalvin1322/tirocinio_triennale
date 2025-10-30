import astra
import numpy as np

def run_fbp_reconstruction(vol_geom, sinogram_id: astra.data2d.create, filter_type: str, use_cuda: bool = True) -> np.ndarray:
    """
    Performs a single FBP (Filtered Back-Projection) reconstruction 
    using ASTRA.

    This function handles creating the ASTRA data structures, 
    configuring the algorithm, running it, and cleaning up the memory.

    Args:
        vol_geom: The ASTRA volume geometry (created with astra.create_vol_geom).
        sinogram_id: The ASTRA ID of the sinogram data 
                       (created with astra.data2d.create).
        filter_type: The name of the filter to use (e.g., 'ram-lak', 
                     'shepp-logan', 'cosine', etc.).
        use_cuda: If True, uses the FBP_CUDA algorithm. If False, 
                  uses 'FBP' (CPU version).

    Returns:
        np.ndarray: The reconstructed image as a NumPy array.
    """
    
    recon_id = astra.data2d.create('-vol', vol_geom, data=0.0)

    algo_type = 'FBP_CUDA' if use_cuda else 'FBP'
    cfg = astra.astra_dict(algo_type)
    cfg['ProjectionDataId'] = sinogram_id
    cfg['ReconstructionDataId'] = recon_id
    cfg['option'] = {"FilterType": filter_type}
    
    algorithm_id = astra.algorithm.create(cfg)

    try:
        astra.algorithm.run(algorithm_id)
        reconstruction = astra.data2d.get(recon_id)
    
    finally:
        astra.data2d.delete(recon_id)
        astra.algorithm.delete(algorithm_id)
        
    return reconstruction