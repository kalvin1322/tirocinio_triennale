import numpy as np
import astra
import torch
from skimage.restoration import denoise_tv_chambolle


def run_admm_tv_reconstruction(sinogram=None, geometry_config=None, iterations=50, rho=1.0, lambda_tv=0.1, initial_image=None):
    """
    Alternating Direction Method of Multipliers for TV regularized CT reconstruction.
    
    Two modes of operation:
    1. Sinogram mode: Uses variable splitting for full reconstruction (requires sinogram and geometry_config)
    2. Image refining mode: Applies TV denoising to an existing image (requires initial_image)
    
    Args:
        sinogram: The measured sinogram data (2D array) - required for mode 1
        geometry_config: Dictionary containing geometry parameters - required for mode 1:
            - image_size: Size of the reconstruction (default: 256)
            - num_angles: Number of projection angles (default: 180)
            - det_spacing: Detector spacing (default: 1.0)
            - angles: Custom angles array (optional)
        iterations: Number of ADMM iterations (default: 50)
        rho: Penalty parameter for ADMM (default: 1.0)
        lambda_tv: TV regularization weight (default: 0.1)
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
        x = img.astype(np.float32)
        x = denoise_tv_chambolle(x, weight=lambda_tv / rho, max_num_iter=iterations)
        
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
    
    # Initialization
    x = np.zeros((image_size, image_size), dtype=np.float32)
    z = np.zeros_like(x)  # Auxiliary variable for TV
    u = np.zeros_like(x)  # Scaled Lagrange multiplier
    
    def solve_x_subproblem(target_z_u, current_x, n_cgls=5):
        """
        Solve x-subproblem using CGLS (Conjugate Gradient Least Squares).
        Minimizes: ||Ax - b||^2 + rho/2 ||x - (z - u)||^2
        """
        # Compute right-hand side: A^T*b + rho*(z - u)
        rec_id, bp_b = astra.create_backprojection(sinogram, proj_id)
        astra.data2d.delete(rec_id)
        rhs = bp_b + rho * target_z_u
        
        # CGLS iterations
        x_sol = current_x.copy()
        
        # Compute initial residual: rhs - (A^T*A*x + rho*x)
        sino_id, sino_x = astra.create_sino(x_sol, proj_id)
        astra.data2d.delete(sino_id)
        rec_id, bp_fp_x = astra.create_backprojection(sino_x, proj_id)
        astra.data2d.delete(rec_id)
        
        r = rhs - (bp_fp_x + rho * x_sol)
        p = r.copy()
        rsold = np.sum(r * r)
        
        for _ in range(n_cgls):
            # Compute Ap = A^T*A*p + rho*p
            sino_id, sino_p = astra.create_sino(p, proj_id)
            astra.data2d.delete(sino_id)
            rec_id, bp_fp_p = astra.create_backprojection(sino_p, proj_id)
            astra.data2d.delete(rec_id)
            
            Ap = bp_fp_p + rho * p
            alpha = rsold / np.sum(p * Ap)
            x_sol = x_sol + alpha * p
            r = r - alpha * Ap
            rsnew = np.sum(r * r)
            
            if np.sqrt(rsnew) < 1e-6:
                break
                
            p = r + (rsnew / rsold) * p
            rsold = rsnew
            
        return x_sol

    for i in range(iterations):
        # 1. Update x (data fidelity + coupling)
        x = solve_x_subproblem(z - u, x)
        
        # 2. Update z (proximal operator of TV)
        # Solves: min lambda*TV(z) + rho/2 ||z - (x + u)||^2
        # Equivalent to TV denoising on (x + u) with weight lambda/rho
        z = denoise_tv_chambolle(x + u, weight=lambda_tv / rho)
        
        # 3. Update u (dual variable)
        u = u + x - z
        
        if i % 10 == 0:
            # Compute residual for monitoring
            sino_id, sino_x = astra.create_sino(x, proj_id)
            astra.data2d.delete(sino_id)
            res_norm = np.linalg.norm(sino_x - sinogram)
            primal_res = np.linalg.norm(x - z)
            print(f"ADMM Iteration {i}/{iterations}, Data residual: {res_norm:.6f}, Primal residual: {primal_res:.6f}")

    # Cleanup ASTRA objects
    astra.projector.delete(proj_id)
    
    return x
