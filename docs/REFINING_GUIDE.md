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
python run.py test \
  --model UNet_V1 \
  --checkpoint experiments/my_exp/trained_models/FBP_UNet_V1.pth \
  --refining FISTA_TV \
  --visualize \
  --num-samples 10
```

#### Benchmark with Refining
```bash
# Compare multiple models with multiple refining methods
python run.py benchmark \
  --experiment-id 20251111_115226 \
  --preprocessing FBP SART \
  --postprocessing UNet_V1 PostProcessNet \
  --refining FISTA_TV CHAMBOLLE_POCK ADMM_TV
```

This will test:
- Each model **without** refining (baseline)
- Each model **with each** refining method
- Show improvement metrics (PSNR Δ, SSIM Δ, MSE Δ)

### Interactive Mode

```bash
python run.py
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

### Expected Results

Refining typically improves:
- **PSNR**: +0.5 to +2.5 dB improvement
- **MSE**: 10-30% reduction
- **SSIM**: Minimal change (0.00 to +0.01)

TV denoising primarily reduces noise and smooths regions while preserving edges, which improves PSNR/MSE more than SSIM.

---

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

Edit `src/utils/refining_registry.py`:

```python
from src.models.my_refining_method import run_my_refining

# Add to imports section

def my_refining_wrapper(iterations=50, lambda_tv=0.1, initial_image=None):
    """
    Wrapper for My Refining method.
    Adapts custom parameters to standard interface.
    """
    return run_my_refining(
        initial_image=initial_image,
        param1=lambda_tv,  # Map standard parameter
        param2=iterations   # Map standard parameter
    )

# Register in the dictionary
REFINING_METHODS = {
    'FISTA_TV': fista_tv_wrapper,
    'CHAMBOLLE_POCK': chambolle_pock_wrapper,
    'ADMM_TV': admm_tv_wrapper,
    'MY_REFINING': my_refining_wrapper,  # Add your method
}
```

### Step 3: Update Configuration (Optional)

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
python run.py test \
  --model UNet_V1 \
  --checkpoint path/to/model.pth \
  --refining MY_REFINING

# Or use Python API
from src.utils.refining_registry import get_refining_method

method = get_refining_method('MY_REFINING')
refined = method(initial_image=model_output)
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
python run.py test \
  --model SimpleResNet \
  --checkpoint path/to/checkpoint.pth \
  --refining FISTA_TV \
  --num-samples 5
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

## Best Practices

### 1. Parameter Tuning

**Lambda (λ_tv)** - Regularization strength:
- **Low (0.01-0.05)**: Minimal smoothing, preserves details
- **Medium (0.1-0.2)**: Balanced noise reduction
- **High (0.5-1.0)**: Aggressive smoothing, may blur edges

**Iterations**:
- **10-30**: Fast, good for testing
- **50-100**: Standard, good quality
- **100+**: Diminishing returns, slower

**Recommendation**: Start with defaults, then tune based on your data.

### 2. When to Use Refining

✅ **Use refining when**:
- Model output has residual noise
- You need extra 0.5-2 dB PSNR improvement
- Edges need better preservation
- You have computational budget for post-processing

❌ **Skip refining when**:
- Model already produces clean output (high PSNR >30 dB)
- Real-time processing is critical
- SSIM is more important than PSNR

### 3. Benchmarking Strategy

```bash
# Test baseline first
python run.py benchmark \
  --preprocessing FBP \
  --postprocessing UNet_V1

# Then add refining to best models
python run.py benchmark \
  --preprocessing FBP \
  --postprocessing UNet_V1 \
  --refining FISTA_TV CHAMBOLLE_POCK
```

### 4. Combining Methods

You can chain preprocessing → model → refining:

```
FBP → UNet_V1 → FISTA_TV
SART → PostProcessNet → CHAMBOLLE_POCK
ADMM_TV → SimpleResNet → ADMM_TV (double TV)
```

### 5. Debugging

If refining fails:
1. Check input dimensions: `print(initial_image.shape)`
2. Verify data range: `print(initial_image.min(), initial_image.max())`
3. Test on single image first
4. Check error logs for tensor/numpy conversion issues

**Common issues**:
- Dimension mismatch → Check squeeze/unsqueeze
- SSIM calculation error → Images have different shapes
- Slow performance → Reduce iterations or use FISTA_TV

---

## Example Workflows

### Workflow 1: Quick Test
```bash
# Test single model with one refining method
python run.py test \
  --model UNet_V1 \
  --checkpoint trained_models/FBP_UNet_V1.pth \
  --refining FISTA_TV \
  --visualize
```

### Workflow 2: Comprehensive Benchmark
```bash
# Compare all preprocessing + models + refining combinations
python run.py benchmark \
  --experiment-id 20251111_115226 \
  --preprocessing FBP SART SIRT \
  --postprocessing UNet_V1 PostProcessNet SimpleResNet \
  --refining FISTA_TV CHAMBOLLE_POCK ADMM_TV
```

This generates a table like:
```
Model                           | Refining       | PSNR  | SSIM  | MSE    | PSNR Δ | SSIM Δ  | MSE Δ
FBP_UNet_V1                    | none           | 18.50 | 0.305 | 0.0173 | -      | -       | -
FBP_UNet_V1                    | FISTA_TV       | 19.79 | 0.305 | 0.0169 | +1.29  | +0.0000 | +0.0004
FBP_UNet_V1                    | CHAMBOLLE_POCK | 18.74 | 0.305 | 0.0166 | +1.13  | +0.0000 | +0.0007
```

### Workflow 3: Parameter Sweep
```python
# Python script for custom parameter testing
from src.utils.refining_registry import get_refining_method

# Test different lambda values
for lambda_val in [0.05, 0.1, 0.15, 0.2]:
    refining = get_refining_method('FISTA_TV')
    refined = refining(initial_image=output, lambda_tv=lambda_val)
    # Calculate and compare metrics
```

---

## Advanced Topics

### Custom Metrics

To add custom metrics to refining evaluation, modify `src/utils/train_test.py`:

```python
# In apply_refining_to_dataset function
refined_results[i] = {
    'mse': mse,
    'psnr': psnr,
    'ssim': ssim_val,
    'custom_metric': your_metric_function(refined_np, original_np)
}
```

### Conditional Refining

Apply refining only to certain samples:

```python
def conditional_refining(model_output, threshold=20.0):
    # Calculate PSNR of model output
    psnr = calculate_psnr(model_output, ground_truth)
    
    # Only refine if PSNR is below threshold
    if psnr < threshold:
        refining = get_refining_method('FISTA_TV')
        return refining(initial_image=model_output)
    else:
        return model_output
```

### Ensemble Refining

Combine multiple methods:

```python
# Average outputs from different methods
methods = ['FISTA_TV', 'CHAMBOLLE_POCK', 'ADMM_TV']
refined_outputs = []

for method_name in methods:
    method = get_refining_method(method_name)
    refined = method(initial_image=model_output)
    refined_outputs.append(refined)

# Average ensemble
final_output = np.mean(refined_outputs, axis=0)
```

---

## Troubleshooting

### Error: "Input images must have the same dimensions"
- **Cause**: SSIM calculation requires same-sized images
- **Solution**: Check squeeze operations, ensure 2D arrays
- **Fix**: Code now handles this automatically (returns SSIM=0 if shapes differ)

### Error: "initial_image is required"
- **Cause**: Refining method called without image
- **Solution**: Always pass `initial_image=your_tensor`

### Slow Performance
- **Cause**: Too many iterations or using ADMM_TV
- **Solution**: 
  - Reduce iterations (try 30 instead of 50)
  - Use FISTA_TV instead of ADMM_TV
  - Process smaller batches

### No Improvement Observed
- **Possible causes**:
  - Lambda too low (increase to 0.15-0.2)
  - Model output already optimal
  - Data characteristics don't benefit from TV
- **Solution**: Try different lambda values, compare methods

---

## References

- **Total Variation**: Rudin, L. I., Osher, S., & Fatemi, E. (1992). Nonlinear total variation based noise removal algorithms.
- **FISTA**: Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems.
- **Chambolle-Pock**: Chambolle, A., & Pock, T. (2011). A first-order primal-dual algorithm for convex problems.
- **ADMM**: Boyd, S., et al. (2011). Distributed optimization and statistical learning via ADMM.

---

## Summary

The refining system provides:
- ✅ Easy integration with existing models
- ✅ Multiple TV-based methods
- ✅ Comprehensive benchmarking
- ✅ Extensible registry pattern
- ✅ Automatic metrics tracking
- ✅ CLI and Python API support

For questions or issues, refer to the main project documentation or create an issue on GitHub.
