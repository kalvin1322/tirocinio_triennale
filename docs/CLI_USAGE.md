# CLI Usage Guide

Complete guide for using the CT Reconstruction Tool via command line.

## Table of Contents
- [Quick Start](#quick-start)
- [Interactive Mode](#interactive-mode)
- [Non-Interactive Commands](#non-interactive-commands)
- [Batch Processing / SLURM](#batch-processing--slurm)
- [Examples](#examples)

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

## Interactive Mode

For beginners and exploratory work, use the interactive mode:

```bash
python run.py interactive
# or simply
python run.py
```

This launches a guided menu-driven interface that walks you through:
- Creating/selecting experiments
- Training models with visual parameter selection
- Testing models with automatic result visualization
- Benchmarking multiple models

**When to use Interactive Mode:**
- 🎯 First time using the tool
- 🔍 Exploring different models and parameters
- 📊 Quick experiments and testing
- 🎓 Learning the workflow

**When to use CLI Mode:**
- 🖥️ Running on HPC clusters (SLURM, PBS)
- 🤖 Automating workflows with scripts
- 🔄 Batch processing multiple experiments
- 📈 Hyperparameter sweeps

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
- `--postprocessing` / `-post`: Model to train (UNet_V1, ThreeL_SSNet) [required]
- `--preprocessing` / `-pre`: Preprocessing method (default: FBP)
- `--epochs`: Number of training epochs (default: 5)
- `--batch-size` / `-b`: Batch size (default: 8)
- `--learning-rate` / `-lr`: Learning rate (default: 1e-4)

**UNet-Specific Options:**
- `--num-encoders`: Number of encoder-decoder pairs (default: 3)
- `--start-channels`: Starting middle channels (default: 64)

**Advanced Options:**
- `--experiment` / `-e`: Use specific experiment (default: current)
- `--geometry` / `-g`: Geometry config name (default: "default")

**Examples:**

```bash
# Basic training with defaults
python run.py train --postprocessing UNet_V1 --epochs 50

# Custom UNet architecture
python run.py train --postprocessing UNet_V1 --num-encoders 4 --start-channels 128 --epochs 100

# Train SSNet with specific experiment
python run.py train --experiment experiment_20251030_120000 --postprocessing ThreeL_SSNet --epochs 75 --learning-rate 0.00005

# High batch size training
python run.py train --postprocessing UNet_V1 --epochs 50 --batch-size 16 --learning-rate 0.0001
```

**Output:**
- Model saved to `experiments/<name>/trained_models/<model_name>.pth`
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
python run.py test --checkpoint FBP_UNet_V1.pth

# Test with visualization
python run.py test --checkpoint FBP_ThreeL_SSNet.pth --visualize --num-samples 20

# Test specific experiment
python run.py test --experiment my_old_experiment --checkpoint FBP_UNet_V1_enc4_ch128.pth --visualize
```

**Output:**
- Results JSON: `experiments/<name>/test_results/test_results_<model>_<timestamp>.json`
- Plots (if enabled): `experiments/<name>/test_results/plots/visualization_*.png`

---

### 4. Benchmark Models

Compare multiple models side-by-side:

```bash
python run.py benchmark \
  --postprocessing UNet_V1,ThreeL_SSNet
```

**Options:**
- `--postprocessing` / `-post`: Comma-separated list of models to compare [required]
- `--preprocessing` / `-pre`: Preprocessing method (default: FBP)
- `--experiment` / `-e`: Use specific experiment (default: current)

**Examples:**

```bash
# Compare two models
python run.py benchmark --postprocessing UNet_V1,ThreeL_SSNet

# Benchmark with specific experiment
python run.py benchmark \
  --experiment my_experiment \
  --postprocessing UNet_V1,ThreeL_SSNet

# Compare three models
python run.py benchmark \
  --postprocessing UNet_V1,ThreeL_SSNet,CustomModel
```

**Output:**
- Benchmark table: `experiments/<name>/benchmarks/benchmark_<timestamp>.txt`
- Results comparison with metrics (PSNR, SSIM, MSE)

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
