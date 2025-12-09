# Refining Methods Guide

## Overview

Refining methods are post-processing techniques applied **after** neural network reconstruction to further improve image quality. This project implements Total Variation (TV) based denoising methods that can be applied to enhance the output of any trained model.

The refining system uses a **registry pattern**, making it easy to add new methods without modifying existing code.

---

## Table of Contents

1. [Available Refining Methods](#available-refining-methods)
2. [Using Refining Methods](#using-refining-methods)
3. [Adding New Refining Methods](#adding-new-refining-methods)
4. [Modifying Existing Methods](#modifying-existing-methods)
5. [Technical Details](#technical-details)
6. [Best Practices](#best-practices)

---

## Available Refining Methods

The following TV-based refining methods are currently available:

### 1. FISTA_TV (Fast Iterative Shrinkage-Thresholding Algorithm)
- **Algorithm**: Fast proximal gradient method with TV regularization
- **Best for**: General purpose denoising with good speed/quality tradeoff
- **Parameters**:
  - `iterations`: Number of iterations (default: 50)
  - `lambda_tv`: TV regularization weight (default: 0.1)
- **Characteristics**: Fast convergence, good for smooth regions

### 2. CHAMBOLLE_POCK
- **Algorithm**: Primal-dual TV denoising (Chambolle-Pock algorithm)
- **Best for**: High-quality denoising with edge preservation
- **Parameters**:
  - `iterations`: Number of iterations (default: 50)
  - `lambda_tv`: TV regularization weight (default: 0.1)
- **Characteristics**: Superior edge preservation, slightly slower

### 3. ADMM_TV (Alternating Direction Method of Multipliers)
- **Algorithm**: ADMM with TV regularization via variable splitting
- **Best for**: Complex denoising problems
- **Parameters**:
  - `iterations`: Number of iterations (default: 50)
  - `lambda_tv`: TV regularization weight (default: 0.1)
  - `rho`: Penalty parameter (default: 1.0)
- **Characteristics**: Robust convergence, handles constraints well

---

## Using Refining Methods

### Command Line Interface

#### Testing with Refining
```bash
# Test a single model with refining
python run.py test --model UNet_V1 --checkpoint experiments/my_exp/trained_models/FBP_UNet_V1.pth --refining FISTA_TV --visualize --num-samples 10
```

#### Benchmark with Refining
```bash
# Compare multiple models with multiple refining methods
python run.py benchmark --experiment-id 20251111_115226 --preprocessing FBP,SART --postprocessing UNet_V1,PostProcessNet --refining FISTA_TV,CHAMBOLLE_POCK,ADMM_TV
```

This will test:
- Each model **without** refining (baseline)
- Each model **with each** refining method
- Show improvement metrics (PSNR Δ, SSIM Δ, MSE Δ)

### Interactive Mode

```bash
python run.py interactive
# Select "Test" or "Benchmark"
# You'll be prompted to select refining methods
```

The interactive mode will ask:
1. Select your trained model
2. Choose refining method (or skip)
3. Configure visualization options

### Python API

```python
from src.utils.refining_registry import get_refining_method
from src.utils.train_test import apply_refining_to_dataset

# Get refining method from registry
refining_model = get_refining_method('FISTA_TV')

# Apply to a single image
refined_image = refining_model(initial_image=model_output)

# Apply to entire dataset
refined_results, stored_outputs = apply_refining_to_dataset(
    model_instance=model,
    refining_model=refining_model,
    test_dataset=test_dataset,
    device=device,
    store_outputs=True
)
```

## Adding New Refining Methods

### Step 1: Create the Refining Function

Create a new file in `src/models/` (e.g., `my_refining_method.py`):

```python
import numpy as np
import torch

def run_my_refining(initial_image=None, param1=0.1, param2=50):
    """
    My custom refining method.
    
    Args:
        initial_image: Image to refine (tensor or numpy array)
        param1: First parameter (default: 0.1)
        param2: Second parameter (default: 50)
    
    Returns:
        Refined image as 2D numpy array
    """
    if initial_image is None:
        raise ValueError("initial_image is required")
    
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
    
    # YOUR REFINING ALGORITHM HERE
    refined = img.astype(np.float32)
    
    # Example: Apply your processing
    for i in range(param2):
        # Your algorithm logic
        refined = your_processing_function(refined, param1)
    
    return refined
```

**Important conventions**:
- Function name must start with `run_`
- Must accept `initial_image` parameter
- Must return 2D numpy array (float32)
- Handle both tensor and numpy inputs
- Properly squeeze batch/channel dimensions

### Step 2: Register the Method

Edit `src/utils/refining_registry.py` to add your method using the `@register_refining` decorator:

```python
from src.models.my_refining_method import run_my_refining

# Add to the end of the file, after existing registrations

@register_refining("MY_REFINING")
def my_refining_wrapper(param1=0.1, param2=50, **kwargs):
    """
    Wrapper for My Refining method.
    Adapts custom parameters to standard interface.
    """
    return run_my_refining(
        param1=param1,
        param2=param2,
        **kwargs
    )
```

**Key points**:
- Use `@register_refining("METHOD_NAME")` decorator above your wrapper function
- Method name will be automatically converted to uppercase in the registry
- The wrapper function must accept `**kwargs` to handle any additional parameters
- Your method is now automatically available in the refining registry

### Step 3: Update Configuration

Edit `configs/models_config.json` to add metadata:

```json
{
  "refining": {
    "MY_REFINING": {
      "name": "My Refining Method",
      "description": "Custom refining using my algorithm",
      "default_params": {
        "iterations": 50,
        "lambda_tv": 0.1
      }
    }
  }
}
```

### Step 4: Test Your Method

```bash
# Test with CLI
python run.py test --model UNet_V1 --checkpoint path/to/model.pth --refining MY_REFINING
```

---

## Modifying Existing Methods

### Changing Default Parameters

Edit the wrapper in `src/utils/refining_registry.py`:

```python
def fista_tv_wrapper(iterations=100, lambda_tv=0.05, initial_image=None):
    """Changed defaults: more iterations, less regularization"""
    return run_fista_tv_reconstruction(
        iterations=iterations,
        lambda_tv=lambda_tv,
        initial_image=initial_image
    )
```

### Modifying Algorithm Behavior

Edit the implementation file (e.g., `src/models/Fista_Tv_recostruction.py`):

```python
def run_fista_tv_reconstruction(sinogram=None, geometry_config=None, 
                                 iterations=50, lambda_tv=0.1, 
                                 lip_const=None, initial_image=None):
    # Image refining mode (post-neural network)
    if initial_image is not None:
        # ... existing tensor conversion code ...
        
        # MODIFY THIS SECTION
        x = img.astype(np.float32)
        
        # Example: Add preprocessing
        x = your_preprocessing(x)
        
        # Apply TV denoising
        x = denoise_tv_chambolle(x, weight=lambda_tv)
        
        # Example: Add postprocessing
        x = your_postprocessing(x)
        
        return x
```

**Testing after modifications**:
```bash
# Always test on a small dataset first
python run.py test --model SimpleResNet --checkpoint path/to/checkpoint.pth --refining FISTA_TV --num-samples 5
```

---

## Technical Details

### How Refining Works

1. **Neural Network Output**: Model produces initial reconstruction
2. **Refining Application**: TV method applied to reduce noise
3. **Metrics Calculation**: Compare refined vs original images
4. **Improvement Tracking**: Calculate Δ metrics (PSNR, SSIM, MSE)

### Data Flow

```
Original Image
     ↓
Sinogram (noisy)
     ↓
FBP Reconstruction
     ↓
Neural Network → Model Output
     ↓
Refining Method → Refined Output
     ↓
Metrics Calculation (vs Original)
```

### Performance Optimization

The refining methods have been optimized:
- **Single Call**: `denoise_tv_chambolle` called once (not in a loop)
- **GPU Inference**: Model runs with `torch.inference_mode()`
- **Memory Management**: Cache cleared every 10 samples
- **Batch Processing**: Efficient handling of multiple images

### Metrics Tracked

For each refining method, the system tracks:
- **Pre-refining metrics**: PSNR, SSIM, MSE (baseline)
- **Post-refining metrics**: PSNR, SSIM, MSE (after refining)
- **Improvements**: Δ PSNR, Δ SSIM, Δ MSE
- **Percentage improvements**: For MSE reduction

---


For questions or issues, refer to the main project documentation or create an issue on GitHub.
