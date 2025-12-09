import numpy as np
import torch
from skimage.restoration import denoise_tv_chambolle

  
def run_chambolle_pock_reconstruction(iterations=50, lambda_tv=0.1, initial_image=None):
    """
    Chambolle-Pock TV denoising for image refining.
    Applies TV regularization to an existing image (post-neural network processing).
    
    Args:
        iterations: Number of iterations (default: 50)
        lambda_tv: TV regularization weight (default: 0.1)
        initial_image: Image to refine (tensor or numpy array)
    
    Returns:
        Refined image as 2D numpy array
    """
    if initial_image is None:
        raise ValueError("initial_image is required for Chambolle-Pock refining")
    
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
    
    # Apply TV denoising using Chambolle's algorithm
    x = denoise_tv_chambolle(img.astype(np.float32), weight=lambda_tv, max_num_iter=iterations)
    return x
