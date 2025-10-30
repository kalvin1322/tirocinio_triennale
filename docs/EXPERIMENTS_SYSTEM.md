# Experiment Management System

## Overview

The tool now supports a **multi-experiment system** that allows you to:
- Create separate configurations for each experiment
- Keep results organized in dedicated folders
- Run parallel experiments without overwriting data
- Easily compare results from different experiments

## Folder Structure

When you create a new experiment, this structure is automatically generated:

```
experiments/
├── experiments_index.yaml          # Index of all experiments
├── experiment_20231030_143022/     # Example: experiment with timestamp
│   ├── experiment_config.yaml      # Experiment configuration
│   ├── trained_models/             # Trained models
│   ├── test_results/               # Test results
│   ├── benchmarks/                 # Benchmark results
│   └── logs/                       # Operation logs
└── experiment_custom_name/         # Example: experiment with custom name
    ├── experiment_config.yaml
    ├── trained_models/
    └── ...
```

## Creating a New Experiment

### Method 1: Through the Interactive Menu

1. Launch the tool in interactive mode:
   ```bash
   python src/cli/main.py interactive
   ```

2. Select "🔬 Create/Select Experiment"

3. Choose "➕ Create new experiment"

4. The wizard will ask for:
   - **Experiment name**: Identifier name (default: `experiment_<timestamp>`)
   - **Description**: Optional experiment description
   - **Training dataset**: Path to training dataset
   - **Test dataset**: Path to test dataset

5. The system will:
   - Create all necessary folders
   - Verify datasets (count images)
   - Save configuration in `experiment_config.yaml`
   - Update the experiments index

### Experiment Configuration

The `experiment_config.yaml` file contains:

```yaml
experiment:
  name: experiment_20231030_143022
  description: Test with UNet and ThreeL_SSNet
  created_at: 2023-10-30T14:30:22.123456
  timestamp: 20231030_143022

datasets:
  train: Mayo_s Dataset/train
  test: Mayo_s Dataset/test
  train_samples: 3305
  test_samples: 327

output_dirs:
  base: experiments/experiment_20231030_143022
  models: experiments/experiment_20231030_143022/trained_models
  results: experiments/experiment_20231030_143022/test_results
  benchmarks: experiments/experiment_20231030_143022/benchmarks
  logs: experiments/experiment_20231030_143022/logs

configs:
  projection_geometry: configs/projection_geometry.json
  models: configs/models_config.json
```

## Selecting an Existing Experiment

1. In the interactive menu, select "🔬 Create/Select Experiment"
2. Choose "📂 Select existing experiment"
3. Select from the list of available experiments

The selected experiment is saved in `.current_experiment` and used for all subsequent operations (training, testing, benchmark).

## Complete Workflow

### 1. Create a New Experiment

```
🔬 Create/Select Experiment
  → ➕ Create new experiment
  → Name: fbp_comparison_experiment
  → Description: Comparison between UNet and ThreeL_SSNet with FBP
```

**Result**: Folder `experiments/fbp_comparison_experiment/` created

### 2. Train Models

```
🚀 Train a new model
  → Preprocessing: FBP
  → Post-processing: UNet_V1
  → Epochs: 50
```

**Result**: Model saved in `experiments/fbp_comparison_experiment/trained_models/FBP_UNet_V1.pth`

### 3. Test Models

```
🧪 Test an existing model
  → Checkpoint: FBP_UNet_V1.pth
  → Dataset: Mayo_s Dataset/test
```

**Result**: Results saved in `experiments/fbp_comparison_experiment/test_results/`

### 4. Run Benchmark

```
📊 Benchmark multiple models
  → Preprocessing: [✓] FBP
  → Post-processing: [✓] UNet_V1, [✓] ThreeL_SSNet
```

**Result**: Comparison saved in `experiments/fbp_comparison_experiment/benchmarks/`

## System Benefits

### ✅ Organization
- Each experiment has its own dedicated folder
- No risk of overwriting previous results
- Easy navigation between different experiments

### ✅ Reproducibility
- Each experiment saves complete configuration
- Dataset paths, parameters, timestamps recorded
- Easy to reproduce a specific experiment

### ✅ Comparison
- Easily compare results from different experiments
- Centralized index in `experiments_index.yaml`
- Same model can be tested with different configurations

### ✅ Flexibility
- Create experiments with custom names or automatic timestamps
- Work on multiple experiments in parallel
- Switch experiment at any time

## Use Cases

### Case 1: Preprocessing Methods Comparison
```
Experiment 1: FBP_only
  → FBP → UNet_V1
  
Experiment 2: SIRT_comparison
  → SIRT → UNet_V1
  
Comparison: Which preprocessing is better?
```

### Case 2: Hyperparameter Optimization
```
Experiment 1: lr_0001
  → FBP → UNet_V1 (lr=0.001)
  
Experiment 2: lr_00001
  → FBP → UNet_V1 (lr=0.0001)
  
Comparison: Which learning rate is optimal?
```

### Case 3: Different Architectures
```
Experiment 1: unet_comparison
  → FBP → UNet_V1
  
Experiment 2: threelssnet_comparison
  → FBP → ThreeL_SSNet
  
Comparison: Which architecture performs better?
```

## Best Practices

### Naming Conventions
- Use descriptive names: `fbp_unet_50epochs` instead of `exp1`
- Include key information in the name or description
- Use underscores to separate words

### Descriptions
- Write clear and complete descriptions
- Include important details: special parameters, modified dataset, etc.
- Add notes about expected results or hypotheses

### Organization
- Create one experiment per significant variation
- Don't overload a single experiment with too many tests
- Delete failed or irrelevant experiments to maintain order

## Experiments Index

The `experiments/experiments_index.yaml` file keeps track of all experiments:

```yaml
experiments:
  - name: experiment_20231030_143022
    description: Initial test with FBP and UNet
    created_at: 2023-10-30T14:30:22.123456
    config_path: experiments/experiment_20231030_143022/experiment_config.yaml
    
  - name: fbp_comparison_experiment
    description: Comparison between UNet and ThreeL_SSNet
    created_at: 2023-10-30T15:45:10.789012
    config_path: experiments/fbp_comparison_experiment/experiment_config.yaml
```

This allows you to:
- Quickly view all experiments
- Find specific configurations
- Navigate between experiments easily
