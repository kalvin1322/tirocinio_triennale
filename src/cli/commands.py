"""
Command implementations for training, testing, and benchmarking
"""
import torch
import sys
import csv
import json
import re
import click
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from datetime import datetime
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.train_test import (train_step, test_step, save_model, save_test_results, 
                                   save_visualization_plots, apply_refining_to_dataset,
                                   compute_refining_metrics, store_model_outputs)
from src.dataloader.CTDataloader import CTDataset
from src.utils.utilities import astra_projection
from src.utils.postprocessing_registry import get_postprocessing_model, list_postprocessing_models
from src.utils.model_params import build_model_params, validate_param, get_model_filename
from src.utils.refining_registry import get_refining_method, get_all_refining_methods
console = Console()


def get_model(model_name: str, **kwargs):
    """Get model instance by name using registry"""
    
    try:
        return get_postprocessing_model(model_name, **kwargs)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        return None


def train_cmd(preprocessing: str, postprocessing: str, dataset: str, epochs: int, 
              batch_size: int, lr: float, output: str, geometry_config: str = "default",
              **model_params):
    """Train a post-processing model"""
    
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
    
    # Prepare metadata
    metadata = {
        'model_name': postprocessing,
        'preprocessing': preprocessing,
        'model_parameters': params,
        'training_parameters': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'loss_function': loss_fn.__class__.__name__,
        },
        'dataset': dataset,
        'geometry_config': geometry_config,
        'timestamp': datetime.now().isoformat(),
        'device': str(device)
    }
    
    console.print(f"[dim]Saving model as: {model_filename}[/dim]")
    save_path = save_model(model_instance, model_save_name, output_path=str(output_path), metadata=metadata)
    
    console.print(f"\n[bold green]✓ Training completed![/bold green]")
    console.print(f"Model saved to: [cyan]{save_path}[/cyan]\n")


def test_cmd(model: str, checkpoint: str, dataset: str, output: str, experiment_name: str = None, 
             visualize: bool = False, num_samples: int = 5, refining: str = None):
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
    
    # Load metadata from JSON file
    from src.utils.model_params import load_checkpoint_metadata
    metadata = load_checkpoint_metadata(checkpoint)
    
    if not metadata or 'model_parameters' not in metadata:
        console.print(f"[red]Error: Metadata file not found for checkpoint {checkpoint}[/red]")
        console.print(f"[yellow]Make sure the .json file exists alongside the .pth checkpoint[/yellow]")
        return
    
    console.print(f"[green]✓ Loaded model parameters from metadata file[/green]")
    model_params = metadata['model_parameters']
    console.print(f"[cyan]Model parameters:[/cyan]")
    for param, value in model_params.items():
        console.print(f"  {param}: {value}")
    console.print()
    
    # Build complete parameters (defaults + loaded from metadata)
    full_params = build_model_params(model, **model_params)
    
    # Load model using registry with extracted parameters
    model_instance = get_model(model, **full_params)
    if model_instance is None:
        return
    
    # Set loss function based on model type
    loss_fn = torch.nn.L1Loss() if 'UNet' in model else torch.nn.MSELoss()
    
    # Load checkpoint
    try:
        model_instance.load_state_dict(torch.load(checkpoint, weights_only=True))
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
    
    # Display pre-refining metrics
    console.print("\n[bold cyan]═══ Pre-Refining Metrics ═══[/bold cyan]")
    console.print(f"Average MSE:  {test_results['loss']:.6f}")
    console.print(f"Average PSNR: {test_results['psnr']:.2f} dB")
    
    # Refining process
    if refining != None and refining != 'SKIP':
        console.print(f"\n[cyan]Applying refining model: {refining}...[/cyan]\n")
        refining_model = get_refining_method(refining)
        if refining_model is None:
            return
        
        # Apply refining using utility function
        refined_results, stored_outputs = apply_refining_to_dataset(
            model_instance=model_instance,
            refining_model=refining_model,
            test_dataset=test_dataset,
            device=device,
            store_outputs=True,
            console=console
        )
        
        # Compute metrics using utility function
        refining_metrics = compute_refining_metrics(refined_results, test_results)
        
        # Display post-refining metrics
        console.print(f"[green]✓ Refining completed![/green]\n")
        console.print("[bold cyan]═══ Post-Refining Metrics ═══[/bold cyan]")
        console.print(f"Average MSE:  {refining_metrics['mse']:.6f} (Δ {refining_metrics['mse_improvement']:+.6f})")
        console.print(f"Average PSNR: {refining_metrics['psnr']:.2f} dB (Δ {refining_metrics['psnr_improvement']:+.2f} dB)")
        
        # Show improvement summary
        console.print("\n[bold yellow]═══ Improvement Summary ═══[/bold yellow]")
        console.print(f"MSE improved by:  {refining_metrics['mse_improvement_percent']:+.2f}%")
        console.print(f"PSNR improved by: {refining_metrics['psnr_improvement']:+.2f} dB\n")
        
        # Update test_results with refined metrics
        test_results.update(refining_metrics)
    else:
        # No refining - just store model outputs for visualization
        stored_outputs = store_model_outputs(model_instance, test_dataset, device)
    
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
        # Use num_samples parameter to determine how many plots to create
        max_viz_samples = min(num_samples, len(stored_outputs))
        console.print(f"\n[cyan]Generating visualization plots for {max_viz_samples} samples...[/cyan]")
        
        # Use stored outputs (already calculated - no recalculation needed!)
        for i in range(max_viz_samples):
            console.print(f"[yellow]Creating plot {i+1}/{max_viz_samples}...[/yellow]", end="\r")
            
            # Get stored data
            data = stored_outputs[i]
            fbp_input = data['fbp_input']
            original_image = data['original_image']
            model_output = data['model_output']
            refined_output = data['refined_output']  # Will be None if no refining was done
            
            # Create sinogram only for visualization (lazy creation)
            try:
                sinogram = astra_projection(original_image)
            except Exception as e:
                console.print(f"\n[yellow]Warning: Could not create sinogram for sample {i}: {e}[/yellow]")
                # Use a placeholder if sinogram creation fails
                sinogram = np.zeros((180, 512), dtype=np.float32)
            
            # Save visualization (with or without refining based on refined_output)
            save_visualization_plots(
                original_image=original_image,
                sinogram=sinogram,
                fbp_reconstruction=fbp_input.numpy().squeeze(),  # Already on CPU
                model_output=model_output,  # Already on CPU
                refined_output=refined_output,  # None if no refining, otherwise refined image
                output_dir=output,
                model_name=model,
                sample_idx=i
            )
        
        console.print(f"\n[green]✓ Visualization plots saved to: {output}/plots/[/green]")
    
    console.print(f"\n[bold green]✓ Testing completed![/bold green]")
    console.print(f"[cyan]Results saved to: {results_file}[/cyan]\n")


def benchmark_cmd(experiment_id, preprocessing, postprocessing, refining=None, dataset="data/Mayo_s Dataset/test", 
                  geometry_config="default", output=None):
    """Benchmark multiple preprocessing-postprocessing combinations including parameter variants"""
    
    # Setup output directory
    experiment_dir = Path("experiments") / f"experiment_{experiment_id}"
    if not experiment_dir.exists():
        console.print(f"[red]Experiment not found: {experiment_id}[/red]")
        return
    
    if output is None:
        output = str(experiment_dir / "benchmarks")
    
    console.print(f"\n[bold magenta]Starting Benchmark[/bold magenta]")
    console.print(f"Preprocessing methods: {', '.join(preprocessing)}")
    console.print(f"Post-processing models: {', '.join(postprocessing)}")
    console.print(f"Dataset: {dataset}\n")
    
    # Find all trained models in the models directory
    models_dir = experiment_dir / "trained_models"
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
    
    # Process refining methods
    refining_methods = list(refining) if refining else []
    if refining_methods:
        console.print(f"[green]Refining methods to test:[/green]")
        for method in refining_methods:
            console.print(f"  • {method}")
        console.print()
    
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for prep, postp, checkpoint_name, checkpoint_path in matching_checkpoints:
        console.print(f"\n[cyan]Evaluating: {checkpoint_name}...[/cyan]")
        
        # Load model and run test to collect metrics
        try:
            # Load metadata from JSON file
            from src.utils.model_params import load_checkpoint_metadata
            metadata = load_checkpoint_metadata(checkpoint_path)
            
            if not metadata or 'model_parameters' not in metadata:
                console.print(f"[red]Error: Metadata file not found for {checkpoint_name}[/red]")
                continue
            
            model_params_from_file = metadata['model_parameters']
            console.print(f"[dim]  Loaded parameters from metadata[/dim]")
            
            # Build complete model parameters (merge with defaults if needed)
            model_params = build_model_params(postp, **model_params_from_file)
            
            # Load model with correct parameters
            model_instance = get_model(postp, **model_params)
            if model_instance is None:
                continue
            
            model_instance.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
            model_instance.to(device)
            model_instance.eval()
            
            # Load test dataset
            test_dataset = CTDataset(image_path=dataset, geometry_config=geometry_config)
            test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            
            # Run test and collect metrics (without refining)
            loss_fn = torch.nn.L1Loss() if 'UNet' in postp else torch.nn.MSELoss()
            metrics = test_step(model_instance, test_dataloader, loss_fn, device)
            
            # Store baseline results (without refining)
            baseline_result = {
                'model_full_name': checkpoint_name,
                'preprocessing': prep,
                'postprocessing': postp,
                'refining': 'none',
                'checkpoint': f"{checkpoint_name}.pth",
                'psnr': metrics.get('psnr', 0.0),
                'ssim': metrics.get('ssim', 0.0),
                'mse': metrics.get('mse', 0.0),
                'test_loss': metrics.get('loss', 0.0)
            }
            results.append(baseline_result)
            
            console.print(f"  [green]Baseline - PSNR: {metrics.get('psnr', 0):.2f} dB | SSIM: {metrics.get('ssim', 0):.4f} | MSE: {metrics.get('mse', 0):.6f}[/green]")
            
            # Test with refining methods if requested
            if refining_methods:
                for refining_method in refining_methods:
                    if refining_method.lower() == 'none':
                        continue
                        
                    console.print(f"  [cyan]Testing with {refining_method} refining...[/cyan]")
                    
                    try:
                        # Get refining model from registry
                        refining_model = get_refining_method(refining_method)
                        if refining_model is None:
                            console.print(f"    [red]Refining method {refining_method} not found[/red]")
                            continue
                        
                        # Apply refining and compute metrics
                        refined_results, _ = apply_refining_to_dataset(
                            model_instance=model_instance,
                            refining_model=refining_model,
                            test_dataset=test_dataset,
                            device=device,
                            store_outputs=False  # Don't store outputs in benchmark
                        )
                        
                        # Compute aggregated metrics (compute_refining_metrics needs pre_refining_metrics)
                        # Calculate averages manually
                        avg_refined_mse = np.mean([v['mse'] for v in refined_results.values()])
                        avg_refined_psnr = np.mean([v['psnr'] for v in refined_results.values()])
                        avg_refined_ssim = np.mean([v['ssim'] for v in refined_results.values()])
                        
                        # Calculate improvements
                        mse_improvement = metrics.get('mse', 0.0) - avg_refined_mse
                        psnr_improvement = avg_refined_psnr - metrics.get('psnr', 0.0)
                        ssim_improvement = avg_refined_ssim - metrics.get('ssim', 0.0)
                        
                        # Store refining results
                        refining_result = {
                            'model_full_name': checkpoint_name,
                            'preprocessing': prep,
                            'postprocessing': postp,
                            'refining': refining_method,
                            'checkpoint': f"{checkpoint_name}.pth",
                            'psnr': avg_refined_psnr,
                            'ssim': avg_refined_ssim,
                            'mse': avg_refined_mse,
                            'test_loss': avg_refined_mse,
                            'psnr_improvement': psnr_improvement,
                            'mse_improvement': mse_improvement,
                            'ssim_improvement': ssim_improvement
                        }
                        results.append(refining_result)
                        
                        console.print(f"    [green]{refining_method} - PSNR: {avg_refined_psnr:.2f} dB "
                                    f"(+{psnr_improvement:.2f}) | "
                                    f"SSIM: {avg_refined_ssim:.4f} | "
                                    f"MSE: {avg_refined_mse:.6f} "
                                    f"({mse_improvement:+.6f})[/green]")
                        
                    except Exception as e:
                        console.print(f"    [red]Error applying {refining_method}: {e}[/red]")
                        continue
            
        except Exception as e:
            console.print(f"[red]Error testing {prep} → {postp}: {e}[/red]")
            continue
    
    # Display comparison table
    if results:
        table = Table(title="Benchmark Results - All Model Variants")
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Refining", style="magenta", no_wrap=True)
        table.add_column("PSNR (dB)", style="yellow", justify="right")
        table.add_column("SSIM", style="yellow", justify="right")
        table.add_column("MSE", style="yellow", justify="right")
        if refining_methods:
            table.add_column("PSNR Δ", style="green", justify="right")
            table.add_column("SSIM Δ", style="green", justify="right")
            table.add_column("MSE Δ", style="green", justify="right")
        
        for result in results:
            row = [
                result['model_full_name'],
                result['refining'],
                f"{result['psnr']:.2f}",
                f"{result['ssim']:.4f}",
                f"{result['mse']:.6f}"
            ]
            
            # Add improvement columns if refining was used
            if refining_methods:
                if result['refining'] != 'none':
                    row.append(f"+{result.get('psnr_improvement', 0):.2f}")
                    row.append(f"{result.get('ssim_improvement', 0):+.4f}")
                    # MSE improvement is positive when MSE decreases (improvement)
                    mse_imp = result.get('mse_improvement', 0)
                    row.append(f"{mse_imp:+.6f}")
                else:
                    row.append("-")
                    row.append("-")
                    row.append("-")
            
            table.add_row(*row)
        
        console.print("\n")
        console.print(table)
        
        # Save results to files
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = output_path / f"benchmark_{timestamp}.csv"
        json_filename = output_path / f"benchmark_{timestamp}.json"
        
        # Save CSV
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['model_full_name', 'preprocessing', 'postprocessing', 'refining', 'checkpoint', 
                         'psnr', 'ssim', 'mse', 'test_loss']
            if refining_methods:
                fieldnames.extend(['psnr_improvement', 'mse_improvement', 'ssim_improvement'])
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        
        # Save JSON (for more detailed analysis if needed)
        benchmark_data = {
            'timestamp': timestamp,
            'date': datetime.now().isoformat(),
            'preprocessing_methods': list(preprocessing),
            'postprocessing_models': list(postprocessing),
            'refining_methods': refining_methods if refining_methods else ['none'],
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


# CLI wrapper for benchmark_cmd
@click.command(name="benchmark")
@click.option('--experiment-id', required=True, help='Experiment ID to use')
@click.option('--preprocessing', multiple=True, help='Preprocessing methods to test (can specify multiple)')
@click.option('--postprocessing', multiple=True, help='Postprocessing models to test (can specify multiple)')
@click.option('--refining', multiple=True, help='Refining methods to apply after model (can specify multiple: FISTA_TV, CHAMBOLLE_POCK, ADMM_TV)')
@click.option('--dataset', default="data/Mayo_s Dataset/test", help='Path to test dataset')
@click.option('--geometry-config', default="default", help='Geometry configuration name (e.g., default, high_resolution)')
@click.option('--output', default=None, help='Output directory for results (default: <experiment_dir>/benchmarks)')
def benchmark_cli(experiment_id, preprocessing, postprocessing, refining, dataset, geometry_config, output):
    """CLI wrapper for benchmark command"""
    benchmark_cmd(
        experiment_id=experiment_id,
        preprocessing=preprocessing,
        postprocessing=postprocessing,
        refining=refining if refining else None,
        dataset=dataset,
        geometry_config=geometry_config,
        output=output
    )
