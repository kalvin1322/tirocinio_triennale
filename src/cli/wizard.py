"""
Setup wizard for experiment configuration
"""
import os
import inquirer
import yaml
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.panel import Panel

console = Console()


def run_wizard():
    """Run experiment configuration wizard"""
    console.clear()
    console.print(Panel.fit(
        "[bold cyan]CT Reconstruction Experiment Setup[/bold cyan]\n"
        "Create a new experiment configuration with dedicated output directories.",
        title="Experiment Wizard"
    ))
    
    console.print("\n[bold]Step 1: Experiment Configuration[/bold]")
    
    # Generate default experiment name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_exp_name = f"experiment_{timestamp}"
    
    # Get project root (where run.py is located)
    project_root = Path.cwd()
    data_folder = project_root / "data"
    
    console.print(f"[dim]Project directory: {project_root}[/dim]")
    console.print(f"[dim]Data folder: {data_folder}[/dim]\n")
    
    # Scan data folder for available datasets
    available_datasets = []
    if data_folder.exists():
        for dataset_dir in data_folder.iterdir():
            if dataset_dir.is_dir():
                # Check if it has train and test subfolders
                train_dir = dataset_dir / "train"
                test_dir = dataset_dir / "test"
                
                train_count = len(list(train_dir.glob("*/*.png"))) if train_dir.exists() else 0
                test_count = len(list(test_dir.glob("*/*.png"))) if test_dir.exists() else 0
                
                if train_dir.exists() and test_dir.exists():
                    available_datasets.append({
                        'name': dataset_dir.name,
                        'path': dataset_dir,
                        'train': train_dir,
                        'test': test_dir,
                        'train_count': train_count,
                        'test_count': test_count
                    })
        
        if available_datasets:
            console.print(f"[green]✓ Found {len(available_datasets)} dataset(s) in data folder:[/green]")
            for ds in available_datasets:
                console.print(f"  • [cyan]{ds['name']}[/cyan] - {ds['train_count']} train, {ds['test_count']} test images")
            console.print()
        else:
            console.print(f"[yellow]⚠ No valid datasets found in {data_folder}[/yellow]")
            console.print(f"[dim]Expected structure: data/<dataset_name>/train/ and data/<dataset_name>/test/[/dim]\n")
    else:
        console.print(f"[yellow]⚠ Data folder not found: {data_folder}[/yellow]")
        console.print(f"[dim]Please create a 'data' folder in the project root and add your datasets[/dim]\n")
    
    # Build questions based on available datasets
    questions = [
        inquirer.Text(
            'experiment_name',
            message="Enter experiment name (will be used for folders and config)",
            default=default_exp_name,
            validate=lambda _, x: len(x) > 0 and x.replace('_', '').replace('-', '').isalnum()
        ),
        inquirer.Text(
            'description',
            message="Enter experiment description (optional)",
            default=""
        ),
    ]
    
    # Add dataset selection if datasets are available
    if available_datasets:
        dataset_choices = [
            (f"{ds['name']} ({ds['train_count']} train, {ds['test_count']} test)", ds) 
            for ds in available_datasets
        ]
        questions.append(
            inquirer.List(
                'selected_dataset',
                message="Select dataset to use",
                choices=dataset_choices,
            )
        )
    else:
        # Fallback: manual path entry
        questions.extend([
            inquirer.Path(
                'train_dataset',
                message="Enter path to training dataset",
                exists=True,
                path_type=inquirer.Path.DIRECTORY,
                default=str(project_root)
            ),
            inquirer.Path(
                'test_dataset',
                message="Enter path to test dataset",
                exists=True,
                path_type=inquirer.Path.DIRECTORY,
                default=str(project_root)
            ),
        ])
    
    answers = inquirer.prompt(questions)
    
    if answers:
        exp_name = answers['experiment_name']
        
        # Get dataset paths based on selection method
        if 'selected_dataset' in answers:
            # User selected from available datasets
            selected_ds = answers['selected_dataset']
            train_path = selected_ds['train']
            test_path = selected_ds['test']
            dataset_name = selected_ds['name']
        else:
            # User entered manual paths
            train_path = Path(answers['train_dataset'])
            test_path = Path(answers['test_dataset'])
            dataset_name = train_path.parent.name
        
        # Verify datasets
        console.print(f"\n[bold]Step 2: Verifying Dataset[/bold]")
        console.print(f"Dataset: [cyan]{dataset_name}[/cyan]")
        console.print(f"Train path: [dim]{train_path}[/dim]")
        console.print(f"Test path: [dim]{test_path}[/dim]\n")
        
        train_images = list(train_path.glob("*/*.png"))
        test_images = list(test_path.glob("*/*.png"))
        
        if len(train_images) == 0:
            console.print(f"[yellow]⚠ Warning: No .png images found in {train_path}[/yellow]")
            console.print(f"[dim]Expected structure: {train_path}/<subfolder>/*.png[/dim]")
        else:
            console.print(f"[green]✓ Found {len(train_images)} training images[/green]")
            
        if len(test_images) == 0:
            console.print(f"[yellow]⚠ Warning: No .png images found in {test_path}[/yellow]")
            console.print(f"[dim]Expected structure: {test_path}/<subfolder>/*.png[/dim]")
        else:
            console.print(f"[green]✓ Found {len(test_images)} test images[/green]")
        
        # Create experiment-specific output directories
        console.print(f"\n[bold]Step 3: Creating Experiment Directories[/bold]")
        
        experiment_base = Path("experiments") / exp_name
        output_dirs = {
            'models': experiment_base / "trained_models",
            'results': experiment_base / "test_results",
            'benchmarks': experiment_base / "benchmarks",
            'logs': experiment_base / "logs"
        }
        
        for name, dir_path in output_dirs.items():
            dir_path.mkdir(parents=True, exist_ok=True)
            console.print(f"[green]✓ Created {dir_path}[/green]")
        
        # Save configuration
        console.print(f"\n[bold]Step 4: Saving Configuration[/bold]")
        
        config = {
            'experiment': {
                'name': exp_name,
                'description': answers.get('description', ''),
                'created_at': datetime.now().isoformat(),
                'timestamp': timestamp
            },
            'datasets': {
                'train': str(train_path),
                'test': str(test_path),
                'train_samples': len(train_images),
                'test_samples': len(test_images)
            },
            'output_dirs': {
                'base': str(experiment_base),
                'models': str(output_dirs['models']),
                'results': str(output_dirs['results']),
                'benchmarks': str(output_dirs['benchmarks']),
                'logs': str(output_dirs['logs'])
            },
            'configs': {
                'projection_geometry': 'configs/projection_geometry.json',
                'models': 'configs/models_config.json'
            }
        }
        
        # Save experiment config in the experiment folder
        config_path = experiment_base / "experiment_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        console.print(f"[green]✓ Configuration saved to {config_path}[/green]")
        
        # Also save a reference in the main experiments directory
        experiments_index = Path("experiments") / "experiments_index.yaml"
        if experiments_index.exists():
            with open(experiments_index, 'r') as f:
                index = yaml.safe_load(f) or {'experiments': []}
        else:
            index = {'experiments': []}
        
        index['experiments'].append({
            'name': exp_name,
            'description': answers.get('description', ''),
            'created_at': datetime.now().isoformat(),
            'config_path': str(config_path)
        })
        
        with open(experiments_index, 'w') as f:
            yaml.dump(index, f, default_flow_style=False, sort_keys=False)
        
        console.print(f"[green]✓ Experiment indexed in {experiments_index}[/green]")
        
        console.print(Panel.fit(
            f"[bold green]Experiment Setup Complete![/bold green]\n\n"
            f"Experiment: [cyan]{exp_name}[/cyan]\n"
            f"Config: [dim]{config_path}[/dim]\n"
            f"Outputs: [dim]{experiment_base}[/dim]\n\n"
            "You can now:\n"
            "• Use this experiment config in training/testing commands\n"
            "• Run multiple experiments with different configurations\n"
            "• Compare results across experiments",
            title="Success"
        ))
    
    console.input("\n[dim]Press Enter to continue...[/dim]")
    
    return config_path if answers else None


def create_experiment_non_interactive(name: str = None, description: str = "", 
                                     train_dataset: str = None, test_dataset: str = None):
    """
    Create experiment configuration non-interactively (for CLI/scripts)
    
    Args:
        name: Experiment name (default: experiment_<timestamp>)
        description: Experiment description
        train_dataset: Path to training dataset
        test_dataset: Path to test dataset
    
    Returns:
        dict: Experiment configuration
    """
    # Generate default name if not provided
    if not name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"experiment_{timestamp}"
    
    # Validate dataset paths
    train_path = Path(train_dataset)
    test_path = Path(test_dataset)
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_dataset}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found: {test_dataset}")
    
    # Count images
    train_images = list(train_path.glob("*/*.png"))
    test_images = list(test_path.glob("*/*.png"))
    
    if len(train_images) == 0:
        raise ValueError(f"No images found in training dataset: {train_dataset}")
    if len(test_images) == 0:
        raise ValueError(f"No images found in test dataset: {test_dataset}")
    
    # Create experiment structure
    experiment_base = Path("experiments") / name
    output_dirs = {
        'models': experiment_base / "trained_models",
        'results': experiment_base / "test_results",
        'benchmarks': experiment_base / "benchmarks",
        'logs': experiment_base / "logs"
    }
    
    for dir_path in output_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create experiment config
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = {
        'experiment': {
            'name': name,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'timestamp': timestamp
        },
        'datasets': {
            'train': str(train_path),
            'test': str(test_path),
            'train_samples': len(train_images),
            'test_samples': len(test_images)
        },
        'output_dirs': {
            'base': str(experiment_base),
            'models': str(output_dirs['models']),
            'results': str(output_dirs['results']),
            'benchmarks': str(output_dirs['benchmarks']),
            'logs': str(output_dirs['logs'])
        },
        'configs': {
            'projection_geometry': 'configs/projection_geometry.json',
            'models': 'configs/models_config.json'
        }
    }
    
    # Save config
    config_path = experiment_base / "experiment_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    # Save as current experiment
    with open('.current_experiment', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    # Update experiments index
    experiments_dir = Path("experiments")
    experiments_index = experiments_dir / "experiments_index.yaml"
    
    if experiments_index.exists():
        with open(experiments_index) as f:
            index = yaml.safe_load(f) or {'experiments': []}
    else:
        index = {'experiments': []}
    
    index['experiments'].append({
        'name': name,
        'description': description,
        'created_at': datetime.now().isoformat(),
        'config_path': str(config_path)
    })
    
    with open(experiments_index, 'w') as f:
        yaml.dump(index, f, default_flow_style=False, sort_keys=False)
    
    return config
