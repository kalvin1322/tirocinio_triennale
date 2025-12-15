import astra
import numpy as np
def prox_tv(x, lam, n_iters=10):
    """
    Proximal operator for Total Variation (Beck & Teboulle's dual algorithm).
    Solves: min_y 0.5||x - y||^2 + lam * TV(y)
    """
    px = np.zeros_like(x)
    py = np.zeros_like(x)
    
    for _ in range(n_iters):
        gx, gy = spatial_gradient(x - divergence(px, py))
        norm = np.sqrt(gx**2 + gy**2)
        factor = np.maximum(1, norm / lam)
        px = (px + (1.0 / (8.0 * lam)) * gx) / factor
        py = (py + (1.0 / (8.0 * lam)) * gy) / factor
        
    return x - divergence(px, py)
def divergence(dx, dy):
    """Calculates the divergence (Backward difference, added -grad)."""
    h, w = dx.shape
    div = np.zeros_like(dx)
    
    # Divergence on x
    div[:, 0] = dx[:, 0]
    div[:, 1:-1] = dx[:, 1:-1] - dx[:, :-1]
    div[:, -1] = -dx[:, -2]
    
    # Divergence on y
    div[0, :] += dy[0, :]
    div[1:-1, :] += dy[1:-1, :] - dy[:-1, :]
    div[-1, :] += -dy[-2, :]
    
    return div
def spatial_gradient(image):
    """Calculates the spatial gradient (Forward difference)."""
    h, w = image.shape
    dx = np.zeros_like(image)
    dy = np.zeros_like(image)
    dx[:, :-1] = image[:, 1:] - image[:, :-1]
    dy[:-1, :] = image[1:, :] - image[:-1, :]
    return dx, dy