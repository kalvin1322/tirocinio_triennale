import astra
import numpy as np

def create_geometry(image: np.ndarray) -> astra.VolumeGeometry:
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

def create_projection_geometry(num_angles: int = 180, detector_width: int = 768, dist_source_origin: int = 1000, dist_origin_detector: int = 500) -> astra.ProjectionGeometry:
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