"""
Projection geometry configuration loader
Loads geometry configurations from JSON file for CT reconstruction
"""
import json
from pathlib import Path
from typing import Optional
import numpy as np
import astra


def load_projection_geometry(geometry_name: str = "default", 
                             config_path: Optional[str] = None) -> dict:
    """
    Load a projection geometry from JSON configuration file
    
    Args:
        geometry_name: Name of the geometry configuration to load
        config_path: Optional path to custom config file (default: configs/projection_geometry.json)
        
    Returns:
        ASTRA projection geometry dictionary
        
    Example:
        >>> proj_geom = load_projection_geometry("default")
        >>> proj_geom = load_projection_geometry("high_resolution")
    """
    if config_path is None:
        config_path = "configs/projection_geometry.json"
    
    config_path = Path(config_path)
    
    # Load JSON configuration
    if not config_path.exists():
        raise FileNotFoundError(
            f"Projection geometry config not found at: {config_path}\n"
            "Please create the config file with geometry definitions."
        )
    
    with open(config_path, 'r') as f:
        geometries = json.load(f)
    
    # Get specific geometry
    if geometry_name not in geometries:
        available = ", ".join(geometries.keys())
        raise ValueError(
            f"Geometry '{geometry_name}' not found in config.\n"
            f"Available geometries: {available}"
        )
    
    config = geometries[geometry_name]
    
    # Generate angles
    angle_start, angle_end = config['angle_range']
    angles = np.linspace(angle_start, angle_end, config['num_angles'], endpoint=False)
    
    # Create ASTRA geometry based on type
    geometry_type = config['geometry_type']
    
    if geometry_type == 'parallel':
        proj_geom = astra.create_proj_geom(
            'parallel',
            config['detector_spacing'],
            config['detector_count'],
            angles
        )
    elif geometry_type in ['fanflat', 'fanbeam']:
        proj_geom = astra.create_proj_geom(
            'fanflat',
            config['detector_spacing'],
            config['detector_count'],
            angles,
            config['source_origin_distance'],
            config['origin_detector_distance']
        )
    else:
        raise ValueError(
            f"Unsupported geometry type: {geometry_type}\n"
            f"Supported types: 'parallel', 'fanflat', 'fanbeam'"
        )
    
    return proj_geom
