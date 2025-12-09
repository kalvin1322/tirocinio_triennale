import numpy as np
import astra
import torch
from skimage.restoration import denoise_tv_chambolle


def run_fista_tv_reconstruction(sinogram=None, geometry_config=None, iterations=50, lambda_tv=0.1, lip_const=None, initial_image=None):
    """
    Fast Iterative Shrinkage-Thresholding Algorithm with TV regularization.
    
    Two modes of operation:
    1. Sinogram mode: Minimizes ||Ax - b||^2 + lambda * TV(x) (requires sinogram and geometry_config)
    2. Image refining mode: Applies TV denoising to an existing image (requires initial_image)
    
    Args:
        sinogram: The measured sinogram data (2D array) - required for mode 1
        geometry_config: Dictionary containing geometry parameters - required for mode 1:
            - image_size: Size of the reconstruction (default: 256)
            - num_angles: Number of projection angles (default: 180)
            - det_spacing: Detector spacing (default: 1.0)
            - angles: Custom angles array (optional)
        iterations: Number of FISTA iterations (default: 50)
        lambda_tv: TV regularization weight (default: 0.1)
        lip_const: Lipschitz constant (auto-estimated if None)
        initial_image: Initial image for refining mode (tensor or numpy array) - required for mode 2
    
    Returns:
        Reconstructed image as 2D numpy array
    """
    # Image refining mode (post-neural network)
    if initial_image is not None:
        # Convert tensor to numpy if needed
        if torch.is_tensor(initial_image):
            if len(initial_image.shape) == 4:  # Batch dimension
                img = initial_image.squeeze().detach().cpu().numpy()
            elif len(initial_image.shape) == 3:  # Channel dimension
                img = initial_image.squeeze().detach().cpu().numpy()
            else:
                img = initial_image.detach().cpu().numpy()
        else:
            img = initial_image
        
        # Apply TV denoising once (denoise_tv_chambolle is already iterative internally)
        # The 'weight' parameter controls the regularization strength
        x = img.astype(np.float32)
        x = denoise_tv_chambolle(x, weight=lambda_tv)
        
        return x
    
    # Sinogram reconstruction mode (original)
    if sinogram is None or geometry_config is None:
        raise ValueError("Either provide (sinogram + geometry_config) or initial_image")
    # Extract geometry parameters
    image_size = geometry_config.get('image_size', 256)
    num_angles = geometry_config.get('num_angles', 180)
    det_spacing = geometry_config.get('det_spacing', 1.0)
    angles = geometry_config.get('angles', np.linspace(0, np.pi, num_angles, endpoint=False))
    
    # Create ASTRA geometries
    proj_geom = astra.create_proj_geom('parallel', det_spacing, sinogram.shape[1], angles)
    vol_geom = astra.create_vol_geom(image_size, image_size)
    
    # Create projector
    proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
    
    # Initialize reconstruction
    x = np.zeros((image_size, image_size), dtype=np.float32)
    t = 1.0
    y = x.copy()
    
    # Estimate Lipschitz constant using power method if not provided
    if lip_const is None:
        print("Estimating Lipschitz constant...")
        x_tmp = np.random.randn(image_size, image_size).astype(np.float32)
        for _ in range(10):
            # Forward projection
            sino_id, sino = astra.create_sino(x_tmp, proj_id)
            astra.data2d.delete(sino_id)
            # Back projection
            rec_id, rec = astra.create_backprojection(sino, proj_id)
            astra.data2d.delete(rec_id)
            x_tmp = rec
            
            n = np.linalg.norm(x_tmp)
            if n > 0:
                x_tmp /= n
        lip_const = n * 1.1  # Safety margin
        print(f"Estimated Lipschitz constant: {lip_const:.4f}")
    
    step_size = 1.0 / lip_const

    for i in range(iterations):
        # Forward projection: A(y)
        sino_id, sino_y = astra.create_sino(y, proj_id)
        astra.data2d.delete(sino_id)
        
        # Residual: A(y) - b
        residual = sino_y - sinogram
        
        # Back projection: A^T * residual
        rec_id, grad = astra.create_backprojection(residual, proj_id)
        astra.data2d.delete(rec_id)
        
        # Gradient descent step
        x_next_uncorr = y - step_size * grad
        
        # Proximal operator (TV denoising)
        x_next = denoise_tv_chambolle(x_next_uncorr, weight=lambda_tv * step_size)
        
        # FISTA momentum update
        t_next = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x_next + ((t - 1) / t_next) * (x_next - x)
        
        x = x_next
        t = t_next
        
        if i % 10 == 0:
            # Compute residual norm for monitoring
            res_norm = np.linalg.norm(residual)
            print(f"FISTA-TV Iteration {i}/{iterations}, Residual norm: {res_norm:.6f}")

    # Cleanup ASTRA objects
    astra.projector.delete(proj_id)
    
    return x
