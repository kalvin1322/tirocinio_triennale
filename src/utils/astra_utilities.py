import astra
import numpy as np

def create_geometry(image: np.ndarray) -> dict:
    """
    Creates an ASTRA volume geometry for a given image.

    Args:
        image: The 2D NumPy array representing the image.
    Returns:
        astra.VolumeGeometry: The ASTRA volume geometry object.
    """
    # Get the image dimensions
    height, width = image.shape

    # Create the ASTRA volume geometry
    vol_geom = astra.create_vol_geom(width, height)

    return vol_geom

def create_projection_geometry(num_angles: int = 180, detector_width: int = 768, dist_source_origin: int = 1000, dist_origin_detector: int = 500) -> dict:
    """
    Creates an ASTRA projection geometry for parallel beam geometry.

    Args:
        num_angles: Number of projection angles.
        detector_size: Size of the detector (number of bins).
    Returns:
        astra.ProjectionGeometry: The ASTRA projection geometry object.
    """
    angles = np.linspace(0, np.pi, num_angles, False)

    proj_geom = astra.create_proj_geom(
        'fanflat',              # type of geometry
        1.0,                    # distance between the centers of two adjacent detector pixels
        detector_width,         # number of detector pixels
        angles,                 # projection angles in radians
        dist_source_origin,     # distance between the source and the center of rotation DSO
        dist_origin_detector    # distance between the center of rotation and the detector array DOD
    )
    return proj_geom
def prox_tv(x, lam, n_iters=10):
    """
    Operatore prossimale per la Total Variation (algoritmo duale di Beck & Teboulle).
    Risolve: min_y 0.5||x - y||^2 + lam * TV(y)
    """
    # Inizializza le variabili duali
    px = np.zeros_like(x)
    py = np.zeros_like(x)
    
    # Gradiente proiettato
    for _ in range(n_iters):
        gx, gy = spatial_gradient(x - divergence(px, py))
        norm = np.sqrt(gx**2 + gy**2)
        factor = np.maximum(1, norm / lam)
        px = (px + (1.0 / (8.0 * lam)) * gx) / factor
        py = (py + (1.0 / (8.0 * lam)) * gy) / factor
        
    return x - divergence(px, py)
def divergence(dx, dy):
    """Calcola la divergenza (Backward difference, aggiunto di -grad)."""
    h, w = dx.shape
    div = np.zeros_like(dx)
    
    # Divergenza su x
    div[:, 0] = dx[:, 0]
    div[:, 1:-1] = dx[:, 1:-1] - dx[:, :-1]
    div[:, -1] = -dx[:, -2]
    
    # Divergenza su y
    div[0, :] += dy[0, :]
    div[1:-1, :] += dy[1:-1, :] - dy[:-1, :]
    div[-1, :] += -dy[-2, :]
    
    return div
def spatial_gradient(image):
    """Calcola il gradiente spaziale (Forward difference)."""
    h, w = image.shape
    dx = np.zeros_like(image)
    dy = np.zeros_like(image)
    dx[:, :-1] = image[:, 1:] - image[:, :-1]
    dy[:-1, :] = image[1:, :] - image[:-1, :]
    return dx, dy