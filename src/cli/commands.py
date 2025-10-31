"""
Command implementations for training, testing, and benchmarking
"""
import torch
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from datetime import datetime
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
              **model_params):
    """Train a post-processing model"""
    from src.utils.model_params import build_model_params, validate_param, get_model_filename
    
    console.print(f"\n[bold green]Starting Training[/bold green]")
    console.print(f"Preprocessing: {preprocessing}")
    console.print(f"Post-processing Model: {postprocessing}")
    console.print(f"Dataset: {dataset}")
    console.print(f"Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}")
    
    # Build model parameters from config
    params = build_model_params(postprocessing, **model_params)
    
    # Display custom parameters if any
    if model_params:
        console.print(f"Custom parameters: {model_params}")
    
    console.print()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Using device: [cyan]{device}[/cyan]\n")
    
    # Create model with parameters from config
    model_instance = get_model(postprocessing, **params)
    if model_instance is None:
        return
    
    console.print(f"[dim]Model created with parameters: {params}[/dim]")
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
    
    # Build model filename with training parameters and model parameters
    model_filename = get_model_filename(preprocessing, postprocessing, epochs=epochs, lr=lr, **params)
    model_save_name = model_filename.replace('.pth', '')  # Remove extension for save_model
    
    console.print(f"[dim]Saving model as: {model_filename}[/dim]")
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
    """Benchmark multiple preprocessing-postprocessing combinations including parameter variants"""
    import csv
    import json
    import re
    
    console.print(f"\n[bold magenta]Starting Benchmark[/bold magenta]")
    console.print(f"Preprocessing methods: {', '.join(preprocessing)}")
    console.print(f"Post-processing models: {', '.join(postprocessing)}")
    console.print(f"Dataset: {dataset}\n")
    
    # Find all trained models in the models directory
    models_dir = Path(output).parent / "trained_models"
    if not models_dir.exists():
        console.print(f"[red]Models directory not found: {models_dir}[/red]")
        return
    
    available_checkpoints = list(models_dir.glob("*.pth"))
    
    # Filter checkpoints based on requested preprocessing and postprocessing
    matching_checkpoints = []
    for checkpoint in available_checkpoints:
        checkpoint_name = checkpoint.stem  # filename without .pth
        
        # Check if matches any requested preprocessing
        for prep in preprocessing:
            # Check if matches any requested postprocessing (including variants)
            for postp in postprocessing:
                # Match patterns like: FBP_UNet_V1, FBP_UNet_V1_enc4, FBP_UNet_V1_enc4_ch128
                pattern = f"^{re.escape(prep)}_{re.escape(postp)}(_.*)?$"
                if re.match(pattern, checkpoint_name):
                    matching_checkpoints.append((prep, postp, checkpoint_name, checkpoint))
                    break
    
    if not matching_checkpoints:
        console.print(f"[yellow]No trained models found matching the criteria.[/yellow]")
        console.print(f"[dim]Looking for: {preprocessing} → {postprocessing}[/dim]")
        console.print(f"\n[cyan]Available models in {models_dir}:[/cyan]")
        for cp in available_checkpoints:
            console.print(f"  • {cp.name}")
        return
    
    console.print(f"[green]Found {len(matching_checkpoints)} model(s) to benchmark:[/green]")
    for prep, postp, checkpoint_name, _ in matching_checkpoints:
        console.print(f"  • {checkpoint_name}")
    console.print()
    
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for prep, postp, checkpoint_name, checkpoint_path in matching_checkpoints:
        console.print(f"\n[cyan]Evaluating: {checkpoint_name}...[/cyan]")
        
        # Load model and run test to collect metrics
        try:
            # Load model
            model_instance = get_model(postp)
            if model_instance is None:
                continue
            
            model_instance.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model_instance.to(device)
            model_instance.eval()
            
            # Load test dataset
            test_dataset = CTDataset(image_path=dataset, geometry_config=geometry_config)
            test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            
            # Run test and collect metrics
            loss_fn = torch.nn.L1Loss() if 'UNet' in postp else torch.nn.MSELoss()
            metrics = test_step(model_instance, test_dataloader, loss_fn, device)
            
            results.append({
                'model_full_name': checkpoint_name,  # Full model name with all parameters
                'preprocessing': prep,
                'postprocessing': postp,
                'checkpoint': f"{checkpoint_name}.pth",
                'psnr': metrics.get('psnr', 0.0),
                'ssim': metrics.get('ssim', 0.0),
                'mse': metrics.get('mse', 0.0),
                'test_loss': metrics.get('loss', 0.0)  # test_step returns 'loss', not 'test_loss'
            })
            
            console.print(f"  [green]PSNR: {metrics.get('psnr', 0):.2f} dB | SSIM: {metrics.get('ssim', 0):.4f} | MSE: {metrics.get('mse', 0):.6f}[/green]")
            
        except Exception as e:
            console.print(f"[red]Error testing {prep} → {postp}: {e}[/red]")
            continue
    
    # Display comparison table
    if results:
        table = Table(title="Benchmark Results - All Model Variants")
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("PSNR (dB)", style="yellow", justify="right")
        table.add_column("SSIM", style="yellow", justify="right")
        table.add_column("MSE", style="yellow", justify="right")
        
        for result in results:
            # Show full model name with parameters
            table.add_row(
                result['model_full_name'],
                f"{result['psnr']:.2f}",
                f"{result['ssim']:.4f}",
                f"{result['mse']:.6f}"
            )
        
        console.print("\n")
        console.print(table)
        
        # Save results to CSV
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = output_path / f"benchmark_{timestamp}.csv"
        json_filename = output_path / f"benchmark_{timestamp}.json"
        
        # Save CSV
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['model_full_name', 'preprocessing', 'postprocessing', 'checkpoint', 'psnr', 'ssim', 'mse', 'test_loss']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        
        # Save JSON (for more detailed analysis if needed)
        benchmark_data = {
            'timestamp': timestamp,
            'date': datetime.now().isoformat(),
            'preprocessing_methods': preprocessing,
            'postprocessing_models': postprocessing,
            'dataset': dataset,
            'total_models_found': len(matching_checkpoints),
            'successful_tests': len(results),
            'results': results
        }
        
        with open(json_filename, 'w', encoding='utf-8') as jsonfile:
            json.dump(benchmark_data, jsonfile, indent=2)
        
        console.print(f"\n[bold green]✓ Benchmark completed![/bold green]")
        console.print(f"Successfully tested {len(results)} model(s)")
        console.print(f"\n[cyan]Results saved to:[/cyan]")
        console.print(f"  • CSV:  {csv_filename}")
        console.print(f"  • JSON: {json_filename}\n")
    else:
        console.print("\n[yellow]No combinations could be tested. Please train models first.[/yellow]\n")
