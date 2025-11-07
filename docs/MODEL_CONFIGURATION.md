# Model Configuration System

## Overview

The CT reconstruction tool now supports a flexible preprocessing and post-processing pipeline configuration system. This allows users to easily configure and test different combinations of preprocessing methods and post-processing models through a simple JSON configuration file.

### Parameter Configuration: Preprocessing vs Postprocessing

**Important Distinction:**

| Aspect | Preprocessing Parameters | Postprocessing Parameters |
|--------|-------------------------|---------------------------|
| **Configuration Location** | `configs/models_config.json` | `configs/model_parameters.json` |
| **CLI Support** | ❌ **NOT configurable via CLI** | ✅ **Dynamically configurable via CLI** |
| **When to Set** | Before starting experiment | During training (CLI or interactive) |
| **Scope** | Dataset-level (affects all models) | Model-specific (per training run) |



## Configuration File

The model configuration is stored in `configs/models_config.json` with the following structure:

```json
{
  "preprocessing": {
    "FBP": {
      "name": "FBP",
      "description": "Filtered Back Projection reconstruction",
      "type": "reconstruction",
      "requires_training": false
    }
  },
  "postprocessing": {
    "UNet_V1": {
      "name": "UNet_V1",
      "description": "U-Net architecture for image denoising",
      "model_class": "UNet_V1",
      "requires_training": true,
      "default_loss": "L1Loss",
      "default_lr": 0.001
    },
    "ThreeL_SSNet": {
      "name": "ThreeL_SSNet",
      "description": "Three-Level Similarity Structure Network",
      "model_class": "ThreeL_SSNet",
      "requires_training": true,
      "default_loss": "MSELoss",
      "default_lr": 0.001
    }
  }
}
```
## Usage

### Training

When training a model, you select:
1. **Preprocessing method**: e.g., FBP
2. **Post-processing model**: e.g., UNet_V1

The system will:
- Apply the preprocessing (FBP) to generate reconstructed images
- Train the post-processing model to enhance those images
- Save the model with naming: `{preprocessing}_{postprocessing}.pth` (e.g., `FBP_UNet_V1.pth`)

Example interactive flow:
```
Select preprocessing method: FBP
Select post-processing model: UNet_V1
Dataset: Mayo_s Dataset/train
Epochs: 50
...
```

### Benchmarking

For benchmarking, you can select:
- **Multiple preprocessing methods**: e.g., FBP
- **Multiple post-processing models**: e.g., UNet_V1, ThreeL_SSNet

The system will:
- Generate all combinations (e.g., FBP→UNet_V1, FBP→ThreeL_SSNet)
- Test each combination that has a trained checkpoint
- Display comparison results in a table

Example:
```
Select preprocessing methods: [✓] FBP
Select post-processing models: [✓] UNet_V1, [✓] ThreeL_SSNet

Testing 2 combinations:
  • FBP → UNet_V1
  • FBP → ThreeL_SSNet
```

## Adding New Models

The system uses a **registry pattern** to make it easy to add new preprocessing methods and postprocessing models without modifying core code.

### Adding a Preprocessing Method

**See complete guide**: **[Adding Preprocessing Methods](./ADDING_PREPROCESSING_METHODS.md)**

---

### Adding a Post-processing Model

**Step 1: Implement your model class**

Create your model in `src/models/MyNewModel.py`:

```python
import torch
import torch.nn as nn

class MyNewModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_layers=3):
        super().__init__()
        # Your model architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # ... more layers
        )
        
    def forward(self, x):
        # Forward pass
        return self.encoder(x)
```

**Step 2: Register your model**

Add to `src/utils/postprocessing_registry.py`:

```python
# Import your model
from models.MyNewModel import MyNewModel

# Register it with a decorator
@register_postprocessing("MyNewModel")
def mynewmodel_factory(in_channels=1, out_channels=1, num_layers=3, **kwargs):
    """MyNewModel factory with custom parameters."""
    return MyNewModel(
        in_channels=in_channels, 
        out_channels=out_channels,
        num_layers=num_layers,
        **kwargs
    )
```

**Step 3: Add configuration**

Add to `configs/models_config.json`:

```json
"postprocessing": {
  "UNet_V1": { ... },
  "ThreeL_SSNet": { ... },
  "MyNewModel": {
    "name": "My New Model",
    "description": "My custom denoising model with advanced features",
    "class": "MyNewModel",
    "in_channels": 1,
    "out_channels": 1
  }
}
```
Add to `configs/models_parameters.json`:
```json
{
  "UNet_V1": { ... },
  "ThreeL_SSNet": { ... },
  "MyNewModel": {
    "description": "A simple residual network (FCN) for post-processing",
    "default_params": {
      "in_channels": 1,
      "out_channels": 1
    },
    "tunable_params": {
      "num_layers": { ... },
      "features": {...},
      ...
    }
  }
}
```

## Helper Functions

The `src/utils/models_config.py` module provides several helper functions:

- `load_models_config()`: Load the entire configuration
- `get_preprocessing_models()`: Get list of available preprocessing methods
- `get_postprocessing_models()`: Get list of available post-processing models
- `get_preprocessing_info(name)`: Get info about a specific preprocessing method
- `get_postprocessing_info(name)`: Get info about a specific post-processing model
- `get_model_combinations(prep_list, postp_list)`: Generate all combinations for benchmarking


## File Structure

```
tirocinio/
├── configs/
│   ├── models_config.json                # Model configurations (metadata)
│   └── projection_geometry.json          # Geometry configurations
├── src/
│   ├── cli/
│   │   ├── commands.py                   # Command implementations (uses registries)
│   │   └── interactive.py                # Interactive menus
│   ├── models/
│   │   ├── UNet_V1.py                    # Post-processing model
│   │   ├── ThreeL_SSNet.py               # Post-processing model
│   │   ├── FBP_reconstruction.py         # Preprocessing method
│   │   ├── SART_reconstruction.py        # Preprocessing method
│   │   └── SIRT_reconstruction.py        # Preprocessing method
│   ├── dataloader/
│   │   └── CTDataloader.py               # Uses preprocessing_registry
│   └── utils/
│       ├── models_config.py              # Configuration helper functions
│       ├── preprocessing_registry.py     # ⭐ Register preprocessing methods here
│       └── postprocessing_registry.py    # ⭐ Register postprocessing models here
├── docs/
│   ├── MODEL_CONFIGURATION.md            # This file
│   └── ADDING_PREPROCESSING_METHODS.md   # Detailed preprocessing guide
└── experiments/
    └── */trained_models/                 # Saved model checkpoints
```

**Key Files for Adding Models:**
- **`preprocessing_registry.py`** - Add new preprocessing methods
- **`postprocessing_registry.py`** - Add new postprocessing models
- **`models_config.json`** - Add metadata
- **`model_parameters.json`** - Add model configuration

### Testing Your Addition
If you want to test youre addition run in a custom python file this script, you should see the model that you've added:
```python
# Check if registered
from src.utils.preprocessing_registry import list_preprocessing_methods
from src.utils.postprocessing_registry import list_postprocessing_models

print("Preprocessing:", list_preprocessing_methods())
print("Postprocessing:", list_postprocessing_models())
```
