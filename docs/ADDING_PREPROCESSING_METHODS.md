# Adding New Preprocessing Methods

This guide explains how to add new preprocessing (reconstruction) methods to the system without modifying the core dataloader code.

## Overview

The system uses a **registry pattern** where preprocessing methods are registered in `src/utils/preprocessing_registry.py`. This allows you to add new methods by simply:
1. Creating your reconstruction function
2. Registering it in the registry
3. Updating the configuration file

**No need to modify CTDataloader!** ✨

---

## Step-by-Step Guide

### Step 1: Create Your Reconstruction Function

Create a new file in `src/models/` with your reconstruction algorithm:

```python
# src/models/SIRT_reconstruction.py

import astra
import numpy as np

def run_sirt_reconstruction(vol_geom, sinogram_id: int, iterations: int = 100) -> np.ndarray:
    """
    Performs SIRT (Simultaneous Iterative Reconstruction Technique) reconstruction.
    
    Args:
        vol_geom: ASTRA volume geometry
        sinogram_id: ASTRA sinogram data ID
        iterations: Number of iterations (default: 100)
    
    Returns:
        np.ndarray: Reconstructed image
    """
    proj_geom = astra.data2d.get_geometry(sinogram_id)
    projector_id = astra.create_projector('linear', proj_geom, vol_geom)
    recon_id = astra.data2d.create('-vol', vol_geom, data=0.0)
    
    cfg = astra.astra_dict('SIRT')
    cfg['ProjectorId'] = projector_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['ReconstructionDataId'] = recon_id
    
    algorithm_id = astra.algorithm.create(cfg)
    
    try:
        astra.algorithm.run(algorithm_id, iterations)
        reconstruction = astra.data2d.get(recon_id)
    finally:
        astra.data2d.delete(recon_id)
        astra.projector.delete(projector_id)
        astra.algorithm.delete(algorithm_id)
    
    return reconstruction
```

**Requirements:**
- Function MUST accept: `vol_geom`, `sinogram_id`
- Function MUST return: `np.ndarray` (the reconstructed image)
- Additional parameters are optional and passed via `**kwargs`

---

### Step 2: Register Your Method

Open `src/utils/preprocessing_registry.py` and add your method at the bottom:

```python
# At the top, add import
from models.SIRT_reconstruction import run_sirt_reconstruction

# At the bottom, add registration (before the comment examples)
@register_preprocessing("SIRT")
def sirt_wrapper(vol_geom, sinogram_id, iterations=100, **kwargs):
    """SIRT (Simultaneous Iterative Reconstruction Technique) wrapper."""
    return run_sirt_reconstruction(
        vol_geom=vol_geom,
        sinogram_id=sinogram_id,
        iterations=iterations
    )
```

**That's it!** Your method is now registered and available system-wide.

---

### Step 3: Update Configuration (Optional but Recommended)

Add your method to `configs/models_config.json`:

```json
{
  "preprocessing": {
    "FBP": { ... },
    "SART": { ... },
    "SIRT": {
      "name": "Simultaneous Iterative Reconstruction Technique",
      "description": "Iterative reconstruction using SIRT algorithm",
      "default_iterations": 100,
      "tunable_params": {
        "iterations": {
          "type": "int",
          "min": 10,
          "max": 500,
          "default": 100,
          "description": "Number of SIRT iterations"
        }
      }
    }
  }
}
```

This enables your method in the CLI and interactive menus.

---


## Advanced: Custom Parameters

Your wrapper can accept any parameters you need:

```python
@register_preprocessing("CUSTOM_METHOD")
def custom_wrapper(vol_geom, sinogram_id, 
                   param1=10, param2="option", enable_feature=True, **kwargs):
    """My custom reconstruction method."""
    return run_custom_reconstruction(
        vol_geom=vol_geom,
        sinogram_id=sinogram_id,
        param1=param1,
        param2=param2,
        enable_feature=enable_feature
    )
```

**Configuration in `models_config.json`:**

```json
{
  "preprocessing": {
    "CUSTOM_METHOD": {
      "name": "Custom Reconstruction Method",
      "description": "My advanced reconstruction with custom parameters",
      "default_param1": 20,
      "default_param2": "other_option",
      "tunable_params": {
        "param1": {
          "type": "int",
          "min": 5,
          "max": 50,
          "default": 20
        },
        "param2": {
          "type": "string",
          "options": ["option", "other_option", "third_option"],
          "default": "other_option"
        },
        "enable_feature": {
          "type": "bool",
          "default": true
        }
      }
    }
  }
}
```



---


## Example: Complete SIRT Implementation

See the complete example files:
- `src/models/SIRT_reconstruction.py` (your algorithm)
- `src/utils/preprocessing_registry.py` (registration)
- `configs/models_config.json` (configuration)

Compare with existing `FBP` and `SART` implementations for reference.
