# CLI Usage Guide

Complete guide for using the CT Reconstruction Tool via command line.

## Table of Contents
- [Quick Start](#quick-start)
- [Important: Parameter Configuration](#important-parameter-configuration)
- [Interactive Mode](#interactive-mode)
- [Non-Interactive Commands](#non-interactive-commands)
- [Batch Processing / SLURM](#batch-processing--slurm)
- [Examples](#examples)

---

## Important: Parameter Configuration

### Preprocessing vs Postprocessing Parameters

⚠️ **Critical Distinction**: 

| Parameter Type | CLI Configurable? | Configuration File | When to Set |
|----------------|-------------------|-------------------|-------------|
| **Preprocessing** (SART/SIRT iterations, FBP filters) | ❌ **NO** | `configs/models_config.json` | Before training |
| **Postprocessing** (UNet encoders, ResNet layers) | ✅ **YES** | `configs/model_parameters.json` | During training (CLI) |

**Why?**
- Preprocessing parameters affect the **dataset** and should be consistent across all models
- Postprocessing parameters are **model-specific** and can vary per training run

**Example Workflow:**

```bash
# Step 1: Configure SIRT iterations in models_config.json
# Edit: configs/models_config.json
# Set: "SIRT": { "default_iterations": 150 }

# Step 2: Train with SIRT preprocessing (uses 150 iterations from config)
python run.py train --preprocessing SIRT --postprocessing SimpleResNet --epochs 50

# Step 3: Customize postprocessing model dynamically via CLI
python run.py train --preprocessing SIRT --postprocessing SimpleResNet --num-layers 5 --features 64 --epochs 50
```

See [Model Configuration Guide](./MODEL_CONFIGURATION.md) for detailed explanation.

---

## Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/kalvin1322/tirocinio_triennale.git
cd tirocinio_triennale

# Install dependencies
conda env create -f environment.yml
conda activate tirocinio


# Prepare your dataset in data/ folder
# Structure: data/Mayo_s Dataset/{train,test}
```


---

## Non-Interactive Commands

### 1. Create Experiment

Create a new experiment configuration:

```bash
python run.py create-experiment --name my_experiment --train-dataset "data/Mayo_s Dataset/train" --test-dataset "data/Mayo_s Dataset/test"
```

**Options:**
- `--name` / `-n`: Experiment name (default: `experiment_<timestamp>`)
- `--description` / `-d`: Optional experiment description
- `--train-dataset`: Path to training data (required)
- `--test-dataset`: Path to test data (required)

**Output:**
- Creates `experiments/<name>/` folder structure
- Saves config to `experiments/<name>/experiment_config.yaml`
- Sets as current experiment in `.current_experiment`

---

### 2. Train Model

Train a postprocessing model:

```bash
python run.py train --postprocessing UNet_V1 --epochs 50 --batch-size 8 --learning-rate 0.0001
```

**Basic Options:**
- `--postprocessing` / `-post`: Model to train (UNet_V1, ThreeL_SSNet, SimpleResNet) [required]
- `--preprocessing` / `-pre`: Preprocessing method (FBP, SART, SIRT) (default: FBP)
- `--epochs`: Number of training epochs (default: 5)
- `--batch-size` / `-b`: Batch size (default: 8)
- `--learning-rate` / `-lr`: Learning rate (default: 1e-4)

**Postprocessing Model-Specific Parameters (Dynamic):**

These parameters are **dynamically accepted** based on the model configuration in `configs/model_parameters.json`:

- **UNet_V1:**
  - `--num-encoders`: Number of encoder-decoder pairs (2-5, default: 3)
  - `--start-channels`: Starting middle channels ([32,64,128,256], default: 64)

- **SimpleResNet:**
  - `--num-layers`: Number of residual layers (2-10, default: 4)
  - `--features`: Features per layer ([16,32,64,128], default: 32)

- **ThreeL_SSNet:**
  - No tunable parameters

**Preprocessing Parameters:**

⚠️ **Important**: Preprocessing parameters (e.g., SART/SIRT iterations) are **NOT configurable via CLI**.  
They must be set in `configs/models_config.json` before training.

Example `models_config.json`:
```json
{
  "preprocessing": {
    "SART": {
      "default_iterations": 50,
      "tunable_params": {
        "iterations": {"default": 50, "min": 10, "max": 200}
      }
    }
  }
}
```

**Advanced Options:**
- `--experiment` / `-e`: Use specific experiment (default: current)
- `--geometry` / `-g`: Geometry config name (default: "default")

**Examples:**

```bash
# Basic training with defaults
python run.py train --postprocessing UNet_V1 --epochs 50
# Output: FBP_UNet_V1_ep50_lr0001.pth

# Custom UNet architecture
python run.py train --postprocessing UNet_V1 --num-encoders 4 --start-channels 128 --epochs 100
# Output: FBP_UNet_V1_ep100_lr0001.pth

# Train SimpleResNet with custom parameters
python run.py train --postprocessing SimpleResNet --num-layers 3 --features 16 --epochs 50 --learning-rate 0.0001
# Output: FBP_SimpleResNet_ep50_lr0001.pth

# Train with SIRT preprocessing (iterations configured in models_config.json)
python run.py train --preprocessing SIRT --postprocessing SimpleResNet --epochs 50
# Output: SIRT_SimpleResNet_ep50_lr0001.pth

# Train SSNet with specific experiment
python run.py train --experiment experiment_20251030_120000 --postprocessing ThreeL_SSNet --epochs 75 --learning-rate 0.00005
# Output: FBP_ThreeL_SSNet_ep75_lr00005.pth
```

**Output:**
- Model saved to `experiments/<name>/trained_models/<model_name>.pth`
- **Filename format**: `<prep>_<postp>[_<arch_params>]_ep<epochs>_lr<learning_rate>.pth`
  - Architecture params (e.g., `enc4_ch128`) only if different from defaults
  - Training params (epochs, lr) **always included**
- Training config saved to `experiments/<name>/logs/`

---

### 3. Test Model

Test a trained model and generate results:

```bash
python run.py test --checkpoint FBP_UNet_V1.pth --visualize --num-samples 10
```

**Options:**
- `--checkpoint` / `-c`: Model checkpoint filename [required]
- `--visualize` / `-v`: Generate visualization plots (default: False)
- `--num-samples` / `-n`: Number of samples to visualize (default: 5)
- `--experiment` / `-e`: Use specific experiment (default: current)

**Examples:**

```bash
# Basic test without visualization
python run.py test --checkpoint FBP_UNet_V1_ep50_lr0001.pth

# Test with visualization
python run.py test --checkpoint FBP_ThreeL_SSNet_ep75_lr00005.pth --visualize --num-samples 20

# Test specific experiment with custom architecture
python run.py test --experiment my_old_experiment --checkpoint FBP_UNet_V1_enc4_ch128_ep100_lr0001.pth --visualize
```

**Output:**
- Results JSON: `experiments/<name>/test_results/test_results_<model>_<timestamp>.json`
- Plots (if enabled): `experiments/<name>/test_results/plots/visualization_*.png`

---

### 4. Benchmark Models

Compare multiple models side-by-side, **including all parameter variants**:

```bash
python run.py benchmark --postprocessing UNet_V1,ThreeL_SSNet
```

**🎯 Important**: This command automatically benchmarks **ALL variants** of the specified models found in your `trained_models` folder. 

For example, if you have trained:
- `FBP_UNet_V1_ep50_lr0001.pth` (default architecture, 50 epochs)
- `FBP_UNet_V1_ep100_lr0001.pth` (default architecture, 100 epochs)
- `FBP_UNet_V1_enc4_ch128_ep50_lr00001.pth` (custom architecture + training)

**All three** will be benchmarked when you specify `--postprocessing UNet_V1`.

**Options:**
- `--postprocessing` / `-post`: Comma-separated list of model types to compare [required]
- `--preprocessing` / `-pre`: Preprocessing method (default: FBP)
- `--experiment` / `-e`: Use specific experiment (default: current)

**Examples:**

```bash
# Benchmark all UNet and SSNet variants
python run.py benchmark --postprocessing UNet_V1,ThreeL_SSNet

# Will automatically find and test ALL trained variants:
#   ✓ FBP_UNet_V1_ep50_lr0001.pth
#   ✓ FBP_UNet_V1_ep100_lr0001.pth
#   ✓ FBP_UNet_V1_enc4_ch128_ep50_lr00001.pth  
#   ✓ FBP_ThreeL_SSNet_ep75_lr00005.pth

# Benchmark specific experiment
python run.py benchmark --experiment my_experiment --postprocessing UNet_V1
```

**Output:**
- **CSV file**: `experiments/<name>/benchmarks/benchmark_<timestamp>.csv`
  ```csv
  model_full_name,preprocessing,postprocessing,checkpoint,psnr,ssim,mse,test_loss
  FBP_UNet_V1_ep50_lr0001,FBP,UNet_V1,FBP_UNet_V1_ep50_lr0001.pth,32.45,0.9234,0.000123,0.045
  FBP_UNet_V1_enc4_ch128_ep50_lr00001,FBP,UNet_V1,FBP_UNet_V1_enc4_ch128_ep50_lr00001.pth,33.12,0.9301,0.000115,0.042
  FBP_ThreeL_SSNet_ep75_lr00005,FBP,ThreeL_SSNet,FBP_ThreeL_SSNet_ep75_lr00005.pth,31.89,0.9156,0.000145,0.052
  ```
  - Shows full model name with parameters in first column
  - Easy to open in Excel/Google Sheets for comparison
  - Compatible with pandas: `pd.read_csv('benchmark.csv')`
  
- **JSON file**: `experiments/<name>/benchmarks/benchmark_<timestamp>.json`
  - Complete metadata and results
  - Includes timestamp, dataset info, and all metrics
  
- **Console table**: Real-time results displayed during execution

---

## Tips & Best Practices

### 1. Experiment Naming
- Use descriptive names: `unet_deep_lr0001` instead of `exp1`
- Include key parameters in name
- Use timestamps for multiple runs

### 2. Resource Management
- **Small experiments**: 5-10 epochs for testing
- **Full training**: 50-100 epochs for production
- **Batch size**: Adjust based on GPU memory (8-16 typical)

### 3. Output Organization
```
experiments/
├── experiment1/
│   ├── trained_models/    # Your models
│   ├── test_results/      # JSON results + plots
│   └── logs/              # Training logs
```

---

## Command Reference

### All Available Commands

```bash
python run.py --help
```

**Commands:**
- `interactive` - Launch interactive guided mode (default if no command)
- `create-experiment` - Create new experiment
- `train` - Train a model
- `test` - Test a model
- `benchmark` - Compare multiple models

### Getting Help

```bash
# Help for specific command
python run.py train --help
python run.py test --help
python run.py create-experiment --help
python run.py benchmark --help
```

---

## Next Steps

- 📖 Read [EXPERIMENTS_SYSTEM.md](EXPERIMENTS_SYSTEM.md) for experiment management details
- 🔧 Check [MODEL_CONFIGURATION.md](MODEL_CONFIGURATION.md) for model configuration
- 💻 See [README.md](../README.md) for overall project documentation
