# CT Reconstruction Training & Benchmarking CLI

A user-friendly command-line tool for training, testing, and benchmarking CT reconstruction models with **multi-experiment support**.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
conda env create -f environment.yml
conda activate tirocinio
```


### 2. Launch Interactive Mode (Recommended)

```bash
python run.py interactive
```

### 3. Create Your First Experiment

When you launch interactive mode, you'll see:
- 🔬 **Create/Select Experiment** - Start here!

Select this option to:
1. Create a new experiment (gets a dedicated folder)
2. Or select an existing experiment to continue working on it

Each experiment gets its own organized folder structure:
```
experiments/
└── your_experiment_name/
    ├── experiment_config.yaml    # Experiment configuration
    ├── trained_models/           # Your trained models
    ├── test_results/             # Test outputs
    ├── benchmarks/               # Benchmark results
    └── logs/                     # Training logs
```

### 4. Start Training/Testing

Once you've created/selected an experiment:
- 🚀 **Train a new model** - Train preprocessing + postprocessing pipelines
- 🧪 **Test an existing model** - Evaluate your models
- 📊 **Benchmark multiple models** - Compare different combinations

## 📖 Usage Modes

### Interactive Mode (Recommended for Beginners) 🌟

Guided step-by-step interface with full experiment management:

```bash
python run.py interactive
```

**Main Menu Features:**
- 🔬 **Create/Select Experiment** - Manage multiple experiments
- 🚀 **Train a new model** - Train preprocessing + postprocessing pipelines
- 🧪 **Test an existing model** - Evaluate trained models with visualization
- 📊 **Benchmark multiple models** - Compare different combinations

### Command-Line Mode (For Automation & HPC) 🚀

Direct commands for scripts, batch jobs, and SLURM:

```bash
# Create experiment
python run.py create-experiment --name my_experiment --train-dataset "data/Mayo_s Dataset/train" --test-dataset "data/Mayo_s Dataset/test"

# Train model
python run.py train --experiment my_experiment --postprocessing UNet_V1 --epochs 50 --batch-size 8

# Test with visualization
python run.py test --experiment my_experiment --checkpoint FBP_UNet_V1.pth --visualize --num-samples 10

# Benchmark multiple models
python run.py benchmark --experiment my_experiment --postprocessing UNet_V1,ThreeL_SSNet
```

**Perfect for:**
- 🖥️ **HPC clusters** (Open OnDemand, SLURM)
- 📜 **Automated scripts** and pipelines
- 🔄 **Batch processing** multiple experiments
- 📊 **Hyperparameter sweeps**

📖 **Full CLI documentation with examples**: [CLI_USAGE.md](docs/CLI_USAGE.md)

## 🎯 Model Pipeline System

The tool uses a **preprocessing + postprocessing** pipeline:

### Preprocessing Methods
- **FBP** (Filtered Back Projection) - Analytical CT reconstruction

### Post-processing Models
- **UNet_V1** - U-Net architecture for image enhancement
- **ThreeL_SSNet** - Three-Level Similarity Structure Network

### Model Combinations
Train and test combinations like:
- `FBP → UNet_V1`
- `FBP → ThreeL_SSNet`

**Configuration**: Edit `configs/models_config.json` to add new models (no code changes needed!)

## 📁 Project Structure

```
tirocinio/
├── src/
│   ├── cli/                      # CLI interface
│   │   ├── main.py               # Main entry point
│   │   ├── interactive.py        # Interactive menus
│   │   ├── commands.py           # Command implementations
│   │   └── wizard.py             # Experiment creation wizard
│   ├── models/                   # Model architectures
│   │   ├── UNet_V1.py            # U-Net model
│   │   └── ThreeL_SSNet.py       # ThreeL-SSNet model
│   ├── dataloader/               # Dataset loaders
│   │   └── CTDataloader.py       # CT dataset with FBP
│   └── utils/                    # Utilities
│       ├── geometry_config.py    # Projection geometry loader
│       ├── models_config.py      # Model configuration loader
│       └── train_test.py         # Training/testing functions
├── configs/                      # Configuration files
│   ├── projection_geometry.json  # CT geometry configurations
│   └── models_config.json        # Model pipeline configurations
├── experiments/                  # Experiment outputs (created on first use)
│   ├── experiments_index.yaml    # Index of all experiments
│   └── experiment_name/          # Individual experiment folder
│       ├── experiment_config.yaml
│       ├── trained_models/
│       ├── test_results/
│       ├── benchmarks/
│       └── logs/
├── Mayo_s Dataset/              # Your CT dataset
│   ├── train/
│   └── test/
├── run.py                       # Quick launcher
├── requirements.txt             # Python dependencies
└── docs/                        # Documentation
    ├── CLI_USAGE.md             # CLI commands & HPC guide
    ├── MODEL_CONFIGURATION.md   # Model config system guide
    └── EXPERIMENTS_SYSTEM.md    # Experiments guide
```

## 💡 Tips

1. **Always Create an Experiment First** - All operations require an active experiment
2. **Use Descriptive Names** - Name experiments clearly (e.g., `fbp_unet_comparison`)
3. **Multiple Experiments** - Run different experiments in parallel without conflicts
4. **Use GPU** - Training is much faster with CUDA
5. **Benchmark Combinations** - Test multiple preprocessing+postprocessing combinations at once
6. **Edit JSON Configs** - Add new models without touching code (see `configs/models_config.json`)
7. **Automate on HPC** - Use CLI commands for SLURM jobs (see [docs/CLI_USAGE.md](docs/CLI_USAGE.md))

## 🔧 Configuration Files

### Experiment Configuration (Auto-generated)

Each experiment gets its own `experiment_config.yaml`:

```yaml
experiment:
  name: my_experiment
  description: Testing UNet vs ThreeL_SSNet
  created_at: 2025-10-30T14:30:22.123456
  
datasets:
  train: Mayo_s Dataset/train
  test: Mayo_s Dataset/test
  train_samples: 3305
  test_samples: 327
  
output_dirs:
  base: experiments/my_experiment
  models: experiments/my_experiment/trained_models
  results: experiments/my_experiment/test_results
  benchmarks: experiments/my_experiment/benchmarks
```

### Model Configuration (User-editable)

#### Model Definitions (`models_config.json`)

Edit `configs/models_config.json` to add new models:

```json
{
  "preprocessing": {
    "FBP": {
      "name": "Filtered Back Projection",
      "description": "Standard FBP reconstruction algorithm",
      "filters": ["ram-lak", "shepp-logan", "cosine", "hamming", "hann"],
      "default_filter": "ram-lak"
    }
  },
  "postprocessing": {
    "UNet_V1": {
      "name": "U-Net V1",
      "description": "U-Net architecture with skip connections for high-quality reconstruction",
      "class": "UNet_V1",
      "in_channels": 1,
      "out_channels": 1
    },
    "ThreeL_SSNet": {
      "name": "Three-Level Squeeze-and-Excitation Network",
      "description": "Lightweight three-level squeeze-and-excitation network for fast enhancement",
      "class": "ThreeL_SSNet"
    }
  }
}
```

#### Model Parameters (`model_parameters.json`)

Customize model hyperparameters in `configs/model_parameters.json`:

```json
{
  "UNet_V1": {
    "description": "U-Net architecture with customizable encoder-decoder pairs",
    "default_params": {
      "in_channels": 1,
      "out_channels": 1,
      "num_encoders": 3,
      "start_middle_channels": 64
    },
    "tunable_params": {
      "num_encoders": {
        "type": "int",
        "min": 2,
        "max": 5,
        "default": 3,
        "description": "Number of encoder-decoder pairs"
      },
      "start_middle_channels": {
        "type": "int",
        "options": [32, 64, 128, 256],
        "default": 64,
        "description": "Starting number of middle channels"
      }
    }
  }
}
```

**Benefits:**
- 🎯 Centralized parameter management
- ✅ Automatic validation (min/max, allowed values)
- 📝 Self-documenting configuration
- 🔧 Easy to add new tunable parameters
- 📛 Automatic model naming with parameters

**Model Naming Examples:**
```bash
# Default parameters → FBP_UNet_V1.pth
python run.py train --postprocessing UNet_V1

# Custom encoders → FBP_UNet_V1_enc4.pth
python run.py train --postprocessing UNet_V1 --num-encoders 4

# Custom encoders + channels → FBP_UNet_V1_enc4_ch128.pth
python run.py train --postprocessing UNet_V1 --num-encoders 4 --start-channels 128
```

## 📊 Example Workflow

### Complete Experiment Flow

```bash
# 1. Launch tool
python run.py interactive

# 2. Create experiment
Select: 🔬 Create/Select Experiment
  → ➕ Create new experiment
  → Name: "unet_vs_threelssnet"
  → Description: "Comparing two post-processing models"
  → Confirm dataset paths

# 3. Train first model (FBP → UNet_V1)
Select: 🚀 Train a new model
  → Preprocessing: FBP
  → Post-processing: UNet_V1
  → Epochs: 50
  → Start training
  
Result: experiments/unet_vs_threelssnet/trained_models/FBP_UNet_V1.pth

# 4. Train second model (FBP → ThreeL_SSNet)
Select: 🚀 Train a new model
  → Preprocessing: FBP
  → Post-processing: ThreeL_SSNet
  → Epochs: 50
  → Start training
  
Result: experiments/unet_vs_threelssnet/trained_models/FBP_ThreeL_SSNet.pth

# 5. Benchmark both models
Select: 📊 Benchmark multiple models
  → Preprocessing: [✓] FBP
  → Post-processing: [✓] UNet_V1, [✓] ThreeL_SSNet
  → Run benchmark

Result: Comparison table showing both models' performance
```



## ❓ FAQ

**Q: Do I need to run the wizard every time?**
A: No! The wizard creates experiments. You can switch between experiments without recreating them.

**Q: Can I have multiple experiments?**
A: Yes! That's the main feature. Create different experiments for different tests/configurations.

**Q: Where are my trained models saved?**
A: In `experiments/{your_experiment_name}/trained_models/`

**Q: How do I switch between experiments?**
A: Use "🔬 Create/Select Experiment" → "📂 Select existing experiment"

**Q: Can I delete old experiments?**
A: Yes, just delete the folder in `experiments/`. They're independent.

**Q: How do I add a new model?**
A: Edit `configs/models_config.json` and add the model class in `src/models/`. See `docs/MODEL_CONFIGURATION.md` for details.

**Q: What's the difference between preprocessing and post-processing?**
A: **Preprocessing** (FBP) reconstructs images from sinograms. **Post-processing** (UNet, ThreeL_SSNet) enhances the reconstructed images.

## 📚 Additional Documentation

- **[CLI Usage Guide](docs/CLI_USAGE.md)** - Complete command-line reference for automation and HPC
- **[Model Configuration Guide](docs/MODEL_CONFIGURATION.md)** - How to add/configure models
- **[Experiments System Guide](docs/EXPERIMENTS_SYSTEM.md)** - Complete guide to the experiments system
- **[projection_geometry_guide](docs/projection_geometry_guide.md)** - Guide for CT machine configuratio
- **[Adding Pre-processing method](docs/ADDING_PREPROCESSING_METHODS.md)** -  Guide to add pre-processing methods
- **[Adding Custom Dataset](data/README.md)** - Guide to add custom dataset

## 📝 License

This project is part of the tirocinio_triennale repository.

## 🤝 Contributing

Feel free to open issues or submit pull requests!
