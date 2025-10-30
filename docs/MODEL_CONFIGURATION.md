# Model Configuration System

## Overview

The CT reconstruction tool now supports a flexible preprocessing and post-processing pipeline configuration system. This allows users to easily configure and test different combinations of preprocessing methods and post-processing models through a simple JSON configuration file.

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

## Pipeline Structure

### Preprocessing Methods
- **FBP (Filtered Back Projection)**: Analytical reconstruction method that converts sinograms to images
- Can be extended with other preprocessing methods (e.g., iterative reconstruction, filtered sinograms)

### Post-processing Models
- **UNet_V1**: Deep learning model for image denoising/enhancement
- **ThreeL_SSNet**: Advanced neural network with similarity structure learning
- Models can be trained to improve reconstruction quality

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

### Adding a Preprocessing Method

Edit `configs/models_config.json`:

```json
"preprocessing": {
  "FBP": { ... },
  "SIRT": {
    "name": "SIRT",
    "description": "Simultaneous Iterative Reconstruction Technique",
    "type": "reconstruction",
    "requires_training": false
  }
}
```

### Adding a Post-processing Model

1. Add the model configuration to `configs/models_config.json`:

```json
"postprocessing": {
  "UNet_V1": { ... },
  "ThreeL_SSNet": { ... },
  "MyNewModel": {
    "name": "MyNewModel",
    "description": "My custom denoising model",
    "model_class": "MyNewModel",
    "requires_training": true,
    "default_loss": "L1Loss",
    "default_lr": 0.0001
  }
}
```

2. Implement the model class in `src/models/MyNewModel.py`:

```python
import torch.nn as nn

class MyNewModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Your model architecture
        
    def forward(self, x):
        # Forward pass
        return x
```

3. Update `src/cli/commands.py` to include the new model:

```python
# In train_cmd function
elif postprocessing == 'MyNewModel':
    model_instance = MyNewModel()
```

## Helper Functions

The `src/utils/models_config.py` module provides several helper functions:

- `load_models_config()`: Load the entire configuration
- `get_preprocessing_models()`: Get list of available preprocessing methods
- `get_postprocessing_models()`: Get list of available post-processing models
- `get_preprocessing_info(name)`: Get info about a specific preprocessing method
- `get_postprocessing_info(name)`: Get info about a specific post-processing model
- `get_model_combinations(prep_list, postp_list)`: Generate all combinations for benchmarking

## Benefits

1. **Easy Configuration**: Add new models by editing JSON, no code changes needed
2. **Flexible Pipelines**: Test different preprocessing + post-processing combinations
3. **Comprehensive Benchmarking**: Automatically test all selected combinations
4. **User-Friendly**: Interactive CLI guides users through model selection
5. **Extensible**: Easy to add new preprocessing methods and post-processing models

## File Structure

```
tirocinio/
├── configs/
│   ├── models_config.json         # Model configurations
│   └── projection_geometry.json   # Geometry configurations
├── src/
│   ├── cli/
│   │   ├── commands.py            # Command implementations
│   │   └── interactive.py         # Interactive menus
│   ├── models/
│   │   ├── UNet_V1.py
│   │   └── ThreeL_SSNet.py
│   └── utils/
│       └── models_config.py       # Configuration helper functions
└── outputs/
    └── trained_models/            # Saved model checkpoints
```

## Example Workflow

1. **Configure models** (edit `configs/models_config.json` if needed)
2. **Train models**:
   ```
   Select: FBP → UNet_V1
   Train for 50 epochs
   Save as: FBP_UNet_V1.pth
   ```
3. **Benchmark**:
   ```
   Select: [✓] FBP
   Select: [✓] UNet_V1, [✓] ThreeL_SSNet
   Test both combinations
   Compare results
   ```
