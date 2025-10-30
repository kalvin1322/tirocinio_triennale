"""
Command implementations for training, testing, and benchmarking
"""
import torch
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.UNet_V1 import UNet_V1
from src.models.ThreeL_SSNet import ThreeL_SSNet
from src.utils.train_test import train_step, test_step, save_model, save_test_results, save_visualization_plots
from src.dataloader.CTDataloader import CTDataset
from torch.utils.data import DataLoader
from src.utils.utilities import astra_projection

console = Console()


def get_model(model_name: str, **kwargs):
    """Get model instance by name"""
    models = {
        'UNet_V1': UNet_V1,
        'ThreeL_SSNet': ThreeL_SSNet,
    }
    
    if model_name not in models:
        console.print(f"[red]Error: Unknown model '{model_name}'[/red]")
        console.print(f"Available models: {', '.join(models.keys())}")
        return None
    
    return models[model_name](**kwargs)


def train_cmd(preprocessing: str, postprocessing: str, dataset: str, epochs: int, 
              batch_size: int, lr: float, output: str, geometry_config: str = "default",
              num_encoders: int = None, start_middle_channels: int = None):
    """Train a post-processing model"""
    console.print(f"\n[bold green]Starting Training[/bold green]")
    console.print(f"Preprocessing: {preprocessing}")
    console.print(f"Post-processing Model: {postprocessing}")
    console.print(f"Dataset: {dataset}")
    console.print(f"Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}")
    
    if num_encoders is not None:
        console.print(f"UNet Encoder Count: {num_encoders}")
    if start_middle_channels is not None:
        console.print(f"UNet Start Mid Channels: {start_middle_channels}")
    console.print()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Using device: [cyan]{device}[/cyan]\n")
    
    # Load model configuration
    from src.utils.models_config import get_postprocessing_info
    model_info = get_postprocessing_info(postprocessing)
    
    # Create model with custom parameters
    if postprocessing == 'UNet_V1':
        unet_params = {'in_channels': 1, 'out_channels': 1}
        
        if num_encoders is not None:
            unet_params['num_encoders'] = num_encoders
        if start_middle_channels is not None:
            unet_params['start_middle_channels'] = start_middle_channels
        
        model_instance = UNet_V1(**unet_params)
        console.print(f"[dim]UNet created with: num_encoders={unet_params.get('num_encoders', 3)}, "
                     f"start_middle_channels={unet_params.get('start_middle_channels', 64)}[/dim]")
    elif postprocessing == 'ThreeL_SSNet':
        model_instance = ThreeL_SSNet()
    else:
        console.print(f"[red]Unknown model: {postprocessing}[/red]")
        return
    
    model_instance.to(device)
    
    # Setup training
    loss_fn = torch.nn.L1Loss() if postprocessing == 'UNet_V1' else torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model_instance.parameters(), lr=lr)
    
    # Load data with preprocessing configuration
    try:
        train_dataset = CTDataset(
            image_path=dataset,
            geometry_config=geometry_config
        )
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    except Exception as e:
        console.print(f"[red]Error loading dataset: {e}[/red]")
        return
    
    # Training loop
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task(f"[cyan]Training {postprocessing}...", total=epochs)
        
        for epoch in range(epochs):
            console.print(f"\n[bold]Epoch {epoch+1}/{epochs}[/bold]")
            
            train_step(
                model=model_instance,
                data_loader=train_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device
            )
            
            progress.update(task, advance=1)
    
    # Save model
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Build model name with parameters if applicable
    model_save_name = f"{preprocessing}_{postprocessing}"
    if postprocessing == 'UNet_V1':
        if num_encoders is not None or start_middle_channels is not None:
            enc = num_encoders if num_encoders is not None else 3
            channels = start_middle_channels if start_middle_channels is not None else 64
            model_save_name = f"{preprocessing}_{postprocessing}_enc{enc}_ch{channels}"
    
    save_path = save_model(model_instance, model_save_name, output_path=str(output_path))
    
    console.print(f"\n[bold green]✓ Training completed![/bold green]")
    console.print(f"Model saved to: [cyan]{save_path}[/cyan]\n")


def test_cmd(model: str, checkpoint: str, dataset: str, output: str, experiment_name: str = None, 
             visualize: bool = False, num_samples: int = 5):
    """Test a trained model"""
    console.print(f"\n[bold blue]Starting Testing[/bold blue]")
    if experiment_name:
        console.print(f"Experiment: {experiment_name}")
    console.print(f"Model: {model}")
    console.print(f"Checkpoint: {checkpoint}")
    console.print(f"Dataset: {dataset}")
    if visualize:
        console.print(f"Visualization: Enabled ({num_samples} samples)\n")
    else:
        console.print()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Using device: [cyan]{device}[/cyan]\n")
    
    # Load model
    if 'UNet' in model:
        model_instance = UNet_V1(in_channels=1, out_channels=1)
        loss_fn = torch.nn.L1Loss()
    elif 'ThreeL_SSNet' in model:
        model_instance = ThreeL_SSNet()
        loss_fn = torch.nn.MSELoss()
    else:
        console.print(f"[red]Unknown model: {model}[/red]")
        return
    
    # Load checkpoint
    try:
        model_instance.load_state_dict(torch.load(checkpoint))
        model_instance.to(device)
        model_instance.eval()
        console.print(f"[green]✓ Model loaded successfully[/green]\n")
    except Exception as e:
        console.print(f"[red]Error loading checkpoint: {e}[/red]")
        return
    
    # Load test data
    try:
        test_dataset = CTDataset(image_path=dataset)
        test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    except Exception as e:
        console.print(f"[red]Error loading dataset: {e}[/red]")
        return
    
    # Test
    console.print("[cyan]Running evaluation...[/cyan]\n")
    test_results = test_step(
        model=model_instance,
        data_loader=test_dataloader,
        loss_fn=loss_fn,
        device=device
    )
    
    # Save results to JSON using utility function
    results_file = save_test_results(
        model_instance=model_instance,
        checkpoint_path=checkpoint,
        dataset_path=dataset,
        dataset_size=len(test_dataset),
        test_metrics=test_results,
        device=device,
        loss_fn=loss_fn,
        output_dir=output,
        model_name=model,
        experiment_name=experiment_name
    )
    
    # Generate visualization plots if requested
    if visualize:
        console.print(f"\n[cyan]Generating visualization plots for {num_samples} samples...[/cyan]")
        import numpy as np
        
        # Get a few samples from the dataset for visualization
        model_instance.eval()
        with torch.inference_mode():
            for i in range(min(num_samples, len(test_dataset))):
                # Get original image and create visualization
                fbp_input, original_image = test_dataset[i]
                
                # Move to device and add batch dimension
                fbp_input_batch = fbp_input.unsqueeze(0).to(device)
                
                # Get model prediction
                model_output = model_instance(fbp_input_batch)
                
                # Recreate sinogram from original image for visualization
                sinogram = astra_projection(original_image)
                
                # Save visualization
                save_visualization_plots(
                    original_image=original_image,
                    sinogram=sinogram,
                    fbp_reconstruction=fbp_input.cpu().numpy().squeeze(),
                    model_output=model_output.cpu(),
                    output_dir=output,
                    model_name=model,
                    sample_idx=i
                )
        
        console.print(f"[green]✓ Visualization plots saved to: {output}/plots/[/green]")
    
    console.print(f"\n[bold green]✓ Testing completed![/bold green]")
    console.print(f"[cyan]Results saved to: {results_file}[/cyan]\n")


def benchmark_cmd(preprocessing: list[str], postprocessing: list[str], dataset: str, 
                 output: str, geometry_config: str = "default"):
    """Benchmark multiple preprocessing-postprocessing combinations"""
    from src.utils.models_config import get_model_combinations
    
    console.print(f"\n[bold magenta]Starting Benchmark[/bold magenta]")
    console.print(f"Preprocessing methods: {', '.join(preprocessing)}")
    console.print(f"Post-processing models: {', '.join(postprocessing)}")
    console.print(f"Dataset: {dataset}\n")
    
    # Generate all combinations
    combinations = get_model_combinations(preprocessing, postprocessing)
    console.print(f"Testing {len(combinations)} combinations:\n")
    
    results = []
    
    for prep, postp in combinations:
        console.print(f"\n[cyan]Evaluating: {prep} → {postp}...[/cyan]")
        
        # Check if checkpoint exists
        checkpoint_name = f"{prep}_{postp}.pth"
        checkpoint_path = Path(output).parent / "trained_models" / checkpoint_name
        
        if not checkpoint_path.exists():
            console.print(f"[yellow]⚠ Checkpoint not found: {checkpoint_path}[/yellow]")
            console.print(f"[dim]Skipping this combination. Train the model first.[/dim]")
            continue
        
        # Run test and collect metrics
        try:
            test_cmd(postp, str(checkpoint_path), dataset, output)
            
            results.append({
                'preprocessing': prep,
                'postprocessing': postp,
                'combination': f"{prep} → {postp}",
                'checkpoint': checkpoint_name
            })
        except Exception as e:
            console.print(f"[red]Error testing {prep} → {postp}: {e}[/red]")
            continue
    
    # Display comparison table
    if results:
        table = Table(title="Benchmark Results - All Combinations")
        table.add_column("Preprocessing", style="cyan")
        table.add_column("→", style="dim")
        table.add_column("Post-processing", style="green")
        table.add_column("Checkpoint", style="yellow")
        
        for result in results:
            table.add_row(
                result['preprocessing'], 
                "→",
                result['postprocessing'],
                result['checkpoint']
            )
        
        console.print("\n")
        console.print(table)
        console.print(f"\n[bold green]✓ Benchmark completed![/bold green]")
        console.print(f"Tested {len(results)} out of {len(combinations)} combinations\n")
    else:
        console.print("\n[yellow]No combinations could be tested. Please train models first.[/yellow]\n")
