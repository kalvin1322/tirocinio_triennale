"""
Main CLI entry point for CT Reconstruction Training & Benchmarking Library
"""
import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from pathlib import Path

from .interactive import run_interactive_mode
from .commands import train_cmd, test_cmd, benchmark_cmd
from .wizard import create_experiment_non_interactive, run_wizard
from src.utils.model_params import validate_param

app = typer.Typer(
    name="ct-benchmark",
    help="CT Reconstruction Training & Benchmarking Tool",
    add_completion=False
)
console = Console()


@app.command()
def interactive():
    """
    🎯 Launch interactive mode with guided setup (Recommended for beginners)
    """
    console.print(Panel.fit(
        "[bold cyan]Welcome to CT Reconstruction Training Tool![/bold cyan]\n"
        "This interactive mode will guide you through the process.",
        title="Interactive Mode"
    ))
    run_interactive_mode()


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def train(
    ctx: typer.Context,
    experiment: str = typer.Option(None, "--experiment", "-e", help="Experiment name (uses .current_experiment if not specified)"),
    preprocessing: str = typer.Option("FBP", "--preprocessing", "-pre", help="Preprocessing method (FBP, SART, SIRT)"),
    postprocessing: str = typer.Option(..., "--postprocessing", "-post", help="Postprocessing model (UNet_V1, ThreeL_SSNet, SimpleResNet)"),
    epochs: int = typer.Option(5, "--epochs", help="Number of training epochs"),
    batch_size: int = typer.Option(8, "--batch-size", "-b", help="Batch size"),
    lr: float = typer.Option(1e-4, "--learning-rate", "-lr", help="Learning rate"),
    geometry_config: str = typer.Option("default", "--geometry", "-g", help="Geometry configuration name")
):
    """
    🚀 Train a postprocessing model
    
    Model-specific parameters are passed dynamically based on the model configuration.
    
    Examples:
      # Basic training
      python run.py train --postprocessing UNet_V1 --epochs 50
      
      # Custom UNet architecture
      python run.py train --postprocessing UNet_V1 --num-encoders 4 --start-channels 128
      
      # Custom SimpleResNet architecture
      python run.py train --postprocessing SimpleResNet --num-layers 3 --features 16
      
      # With specific preprocessing
      python run.py train --preprocessing SIRT --postprocessing SimpleResNet --epochs 10
    """
    # Load experiment config
    if experiment:
        exp_config_path = Path(f"experiments/{experiment}/experiment_config.yaml")
    else:
        exp_config_path = Path(".current_experiment")
    
    if not exp_config_path.exists():
        console.print(f"[red]Error: Experiment config not found at {exp_config_path}[/red]")
        console.print("[yellow]Create an experiment first using 'interactive' mode or specify --experiment[/yellow]")
        raise typer.Exit(1)
    
    with open(exp_config_path) as f:
        exp_config = yaml.safe_load(f)
    
    # Parse extra arguments for model-specific parameters
    model_params = {}
    extra_args = ctx.args
    i = 0
    while i < len(extra_args):
        arg = extra_args[i]
        if arg.startswith('--'):
            param_name = arg[2:].replace('-', '_')  # Convert --num-layers to num_layers
            if i + 1 < len(extra_args) and not extra_args[i + 1].startswith('--'):
                param_value = extra_args[i + 1]
                # Try to convert to appropriate type
                try:
                    # Try int first
                    model_params[param_name] = int(param_value)
                except ValueError:
                    try:
                        # Try float
                        model_params[param_name] = float(param_value)
                    except ValueError:
                        # Keep as string
                        model_params[param_name] = param_value
                i += 2
            else:
                i += 1
        else:
            i += 1
    
    # Display parsed model parameters
    if model_params:
        console.print(f"\n[cyan]Model-specific parameters:[/cyan]")
        for key, value in model_params.items():
            console.print(f"  {key}: {value}")
        
        # Validate parameters
        console.print(f"\n[cyan]Validating parameters...[/cyan]")
        for param_name, param_value in model_params.items():
            is_valid, error_msg = validate_param(postprocessing, param_name, param_value)
            if not is_valid:
                console.print(f"[red]✗ Validation failed for {param_name}: {error_msg}[/red]")
                raise typer.Exit(1)
            else:
                console.print(f"[green]✓ {param_name} = {param_value}[/green]")
    
    # Call train command
    train_cmd(
        preprocessing=preprocessing,
        postprocessing=postprocessing,
        dataset=exp_config['datasets']['train'],
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        output=exp_config['output_dirs']['models'],
        geometry_config=geometry_config,
        **model_params
    )


@app.command()
def test(
    experiment: str = typer.Option(None, "--experiment", "-e", help="Experiment name (uses .current_experiment if not specified)"),
    checkpoint: str = typer.Option(..., "--checkpoint", "-c", help="Model checkpoint filename (e.g., FBP_UNet_V1.pth)"),
    visualize: bool = typer.Option(False, "--visualize", "-v", help="Generate visualization plots"),
    num_samples: int = typer.Option(5, "--num-samples", "-n", help="Number of samples to visualize")
):
    """
    🧪 Test a trained model
    
    Examples:
      # Basic test
      python main.py test --checkpoint FBP_UNet_V1.pth
      
      # Test with visualization
      python main.py test -c FBP_ThreeL_SSNet.pth --visualize --num-samples 10
      
      # Specify experiment
      python main.py test -e my_experiment -c FBP_UNet_V1.pth
    """
    # Load experiment config
    if experiment:
        exp_config_path = Path(f"experiments/{experiment}/experiment_config.yaml")
    else:
        exp_config_path = Path(".current_experiment")
    
    if not exp_config_path.exists():
        console.print(f"[red]Error: Experiment config not found at {exp_config_path}[/red]")
        raise typer.Exit(1)
    
    with open(exp_config_path) as f:
        exp_config = yaml.safe_load(f)
    
    # Build full checkpoint path
    checkpoint_path = Path(exp_config['output_dirs']['models']) / checkpoint
    if not checkpoint_path.exists():
        console.print(f"[red]Error: Checkpoint not found at {checkpoint_path}[/red]")
        raise typer.Exit(1)
    
    # Extract model name from checkpoint
    model_name = checkpoint.replace('.pth', '')
    
    # Call test command
    test_cmd(
        model=model_name,
        checkpoint=str(checkpoint_path),
        dataset=exp_config['datasets']['test'],
        output=exp_config['output_dirs']['results'],
        experiment_name=exp_config['experiment']['name'],
        visualize=visualize,
        num_samples=num_samples
    )


@app.command()
def create_experiment(
    name: str = typer.Option(None, "--name", "-n", help="Experiment name (default: experiment_<timestamp>)"),
    description: str = typer.Option("", "--description", "-d", help="Experiment description"),
    train_dataset: str = typer.Option(..., "--train-dataset", help="Path to training dataset"),
    test_dataset: str = typer.Option(..., "--test-dataset", help="Path to test dataset")
):
    """
    � Create a new experiment
    
    Example:
      python main.py create-experiment \\
        --name my_experiment \\
        --description "Testing UNet variants" \\
        --train-dataset "data/Mayo_s Dataset/train" \\
        --test-dataset "data/Mayo_s Dataset/test"
    """
    exp_config = create_experiment_non_interactive(
        name=name,
        description=description,
        train_dataset=train_dataset,
        test_dataset=test_dataset
    )
    
    console.print(f"\n[green]✓ Experiment created: {exp_config['experiment']['name']}[/green]")
    console.print(f"[cyan]Location: {exp_config['output_dirs']['base']}[/cyan]\n")


@app.command()
def benchmark(
    experiment: str = typer.Option(None, "--experiment", "-e", help="Experiment name (uses .current_experiment if not specified)"),
    preprocessing: str = typer.Option("FBP", "--preprocessing", "-pre", help="Preprocessing methods (comma-separated or single)"),
    postprocessing: str = typer.Option(..., "--postprocessing", "-post", help="Postprocessing models to compare (comma-separated, e.g., 'UNet_V1,ThreeL_SSNet')")
):
    """
    📊 Benchmark multiple models
    
    Examples:
      # Compare two models
      python main.py benchmark --postprocessing UNet_V1,ThreeL_SSNet
      
      # Benchmark specific experiment
      python main.py benchmark -e my_experiment --postprocessing UNet_V1,ThreeL_SSNet
    """
    
    # Load experiment config
    if experiment:
        exp_config_path = Path(f"experiments/{experiment}/experiment_config.yaml")
    else:
        exp_config_path = Path(".current_experiment")
    
    if not exp_config_path.exists():
        console.print(f"[red]Error: Experiment config not found at {exp_config_path}[/red]")
        console.print("[yellow]Create an experiment first using 'interactive' mode or specify --experiment[/yellow]")
        raise typer.Exit(1)
    
    with open(exp_config_path) as f:
        exp_config = yaml.safe_load(f)
    
    # Parse preprocessing and postprocessing
    preprocessing_list = [p.strip() for p in preprocessing.split(',')]
    postprocessing_list = [p.strip() for p in postprocessing.split(',')]
    
    # Call benchmark command
    benchmark_cmd(
        preprocessing=preprocessing_list,
        postprocessing=postprocessing_list,
        dataset=exp_config['datasets']['test'],
        output=exp_config['output_dirs']['benchmarks']
    )


@app.command()
def wizard():
    """
    🧙 Launch setup wizard for first-time configuration
    """
    run_wizard()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """
    CT Reconstruction Training & Benchmarking Tool
    
    Run without arguments to see available commands.
    Use 'interactive' mode for guided experience.
    """
    if ctx.invoked_subcommand is None:
        console.print(Panel.fit(
            "[bold green]CT Reconstruction Training Tool[/bold green]\n\n"
            "Available modes:\n"
            "  • [cyan]interactive[/cyan] - Guided mode (recommended)\n"
            "  • [cyan]create-experiment[/cyan] - Create new experiment\n"
            "  • [cyan]train[/cyan] - Train a model\n"
            "  • [cyan]test[/cyan] - Test a model\n"
            "  • [cyan]benchmark[/cyan] - Compare models\n\n"
            "Run [yellow]python run.py --help[/yellow] for more info",
            title="Welcome"
        ))
        console.print("\n[dim]Tip: Run 'ct-benchmark interactive' to get started[/dim]\n")


if __name__ == "__main__":
    app()
