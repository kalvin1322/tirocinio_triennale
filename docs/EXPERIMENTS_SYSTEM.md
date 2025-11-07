# Experiment Management System

## Overview

The tool supports a **multi-experiment system** that allows you to:
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



## Complete Workflow

### 1. Create a New Experiment

```
🔬 Create/Select Experiment
  → ➕ Create new experiment
  → Name: fbp_comparison_experiment
  → Description: Comparison between UNet and ThreeL_SSNet with FBP
```

**The experiment** folder `experiments/fbp_comparison_experiment/` contains:

* **Trained models** that are saved in `experiments/fbp_comparison_experiment/trained_models/`

* **Test results** that are saved in `experiments/fbp_comparison_experiment/test_results/`

* **Benchmark results** that are saved in `experiments/fbp_comparison_experiment/benchmarks/`


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
