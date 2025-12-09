from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import torch
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

def save_model(model: torch.nn.Module, model_name: str = None, output_path: str = None, metadata: dict = None):
    """Saves a PyTorch model to the specified directory along with metadata.
    
    Args:
        model: The PyTorch model to save
        model_name: Optional custom name. If None, uses the model's class name
        output_path: Directory where to save the model. If None, uses 'outputs/trained_models'
        metadata: Optional dictionary with model parameters and training info
    """
    # 1. Create model directory
    if output_path is None:
        MODEL_PATH = Path("outputs/trained_models")
    else:
        MODEL_PATH = Path(output_path)
    
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    # 2. Get model name from class if not provided
    if model_name is None:
        model_name = model.__class__.__name__
    
    # Add .pth extension if not present
    if not model_name.endswith('.pth'):
        model_name = f"{model_name}.pth"
    
    # 3. Create model save path
    MODEL_SAVE_PATH = MODEL_PATH / model_name
    
    # 4. Save the model state dict
    print(f"Saving the model to: {MODEL_SAVE_PATH}...")
    torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)
    print(f"Model saved successfully as '{model_name}'!")
    
    # 5. Save metadata as JSON if provided
    if metadata is not None:
        metadata_path = MODEL_SAVE_PATH.with_suffix('.json')
        print(f"Saving metadata to: {metadata_path}...")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved successfully!")
    
    return MODEL_SAVE_PATH

def save_test_results(model_instance: torch.nn.Module, checkpoint_path: str, 
                     dataset_path: str, dataset_size: int, test_metrics: dict,
                     device: str, loss_fn: torch.nn.Module, output_dir: str,
                     model_name: str = None, experiment_name: str = None):
    """Saves test results to a JSON file.
    
    Args:
        model_instance: The PyTorch model that was tested
        checkpoint_path: Path to the model checkpoint file
        dataset_path: Path to the test dataset
        dataset_size: Number of samples in the test dataset
        test_metrics: Dictionary containing test metrics (loss, ssim, psnr)
        device: Device used for testing (cuda/cpu)
        loss_fn: Loss function used
        output_dir: Directory where to save the results
        model_name: Optional model name (if None, extracted from checkpoint)
        experiment_name: Optional experiment name to include in results
    
    Returns:
        Path: Path to the saved results file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create results dictionary with all info
    results = {
        'timestamp': datetime.now().isoformat(),
        'experiment': experiment_name,
        'model': {
            'name': model_name or Path(checkpoint_path).stem,
            'checkpoint': checkpoint_path,
            'architecture': model_instance.__class__.__name__
        },
        'dataset': {
            'path': dataset_path,
            'num_samples': dataset_size
        },
        'metrics': test_metrics,
        'device': str(device),
        'loss_function': loss_fn.__class__.__name__
    }
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = Path(checkpoint_path).stem
    results_file = output_path / f"test_results_{checkpoint_name}_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Test results saved to: {results_file}")
    return results_file

def save_visualization_plots(original_image: torch.Tensor, sinogram: np.ndarray,
                            fbp_reconstruction: np.ndarray, model_output: torch.Tensor,
                            output_dir: str, model_name: str, sample_idx: int = 0, 
                            refined_output: torch.Tensor = None):
    """Saves visualization plots comparing original, sinogram, FBP, model and optionally refined reconstruction.
    
    Args:
        original_image: Original ground truth image (torch.Tensor)
        sinogram: Sinogram of the image (numpy array)
        fbp_reconstruction: FBP reconstruction (numpy array)
        model_output: Model output reconstruction (torch.Tensor)
        output_dir: Base directory where to save the plots
        model_name: Name of the model for the filename
        sample_idx: Index of the sample being visualized
        refined_output: Optional refined output (torch.Tensor), if None will show 4 plots instead of 5
    
    Returns:
        Path: Path to the saved plot file
    """
    # Create plots subdirectory
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert tensors to numpy arrays for plotting
    if torch.is_tensor(original_image):
        if len(original_image.shape) == 3:  # (C, H, W)
            original_np = original_image[0].cpu().numpy().squeeze()
        else:
            original_np = original_image.cpu().numpy().squeeze()
    else:
        original_np = original_image
    
    if torch.is_tensor(model_output):
        model_output_np = model_output.squeeze().cpu().numpy()
    else:
        model_output_np = model_output
    
    # Determine number of subplots based on whether refining was used
    num_plots = 5 if refined_output is not None else 4
    
    # Create the figure
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    
    # Original image
    axes[0].imshow(original_np, cmap='gray')
    axes[0].set_title("1. Immagine Originale (Ground Truth)")
    axes[0].axis('off')
    
    # Sinogram
    axes[1].imshow(sinogram, cmap='gray', aspect='auto')
    axes[1].set_title("2. Sinogramma")
    axes[1].set_xlabel("Angolo di proiezione (gradi)")
    axes[1].set_ylabel("Posizione del sensore")
    
    # FBP reconstruction
    axes[2].imshow(fbp_reconstruction, cmap='gray')
    axes[2].set_title("3. Ricostruzione FBP (Input)")
    axes[2].axis('off')
    
    # Model reconstruction
    axes[3].imshow(model_output_np, cmap='gray')
    axes[3].set_title(f"4. Ricostruzione {model_name} (Output)")
    axes[3].axis('off')
    
    # Refined output (if available)
    if refined_output is not None:
        if torch.is_tensor(refined_output):
            refined_output_np = refined_output.squeeze().cpu().numpy()
        else:
            refined_output_np = refined_output
        
        axes[4].imshow(refined_output_np, cmap='gray')
        axes[4].set_title("5. Output Refined (Post-Processing)")
        axes[4].axis('off')
    
    plt.tight_layout()
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    refined_suffix = "_refined" if refined_output is not None else ""
    plot_filename = plots_dir / f"visualization_{model_name}_sample{sample_idx}{refined_suffix}_{timestamp}.png"
    
    # Save the figure
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)  # Close to free memory
    
    return plot_filename

def train_step(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    """Performs a training with model trying to learn on data_loader."""
    
    ### training 
    train_loss = 0
    model.train()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    # Add a loop to loop through the training batches
    for (X, y) in data_loader:
        # Put data on target device
        X, y = X.to(device), y.to(device)
        
        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate the loss (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() # accumulate train los
        ssim_metric.update(y_pred, y)
        psnr_metric.update(y_pred, y)
        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()


    # Divide total train loss and accuracy by length of train dataloader
    train_loss /= len(data_loader)
    final_ssim = ssim_metric.compute()
    final_psnr = psnr_metric.compute()
    print(f"Train loss: {train_loss:.5f} | Train SSIM: {final_ssim:.4f} | Train PSNR: {final_psnr:.2f} dB")

def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device):
    
    """Performs a testing loop step on model going over data_loader.
    
    Returns:
        dict: Dictionary containing test metrics (loss, ssim, psnr)
    """

    test_loss = 0
    mse_total = 0
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            # 1. Forward pass
            test_pred = model(X)

            # 2. Calculate the loss
            test_loss += loss_fn(test_pred, y).item()

            # 3. Calculate MSE
            mse_total += torch.nn.functional.mse_loss(test_pred, y).item()

            # 4. Calculate metrics
            ssim_metric.update(test_pred, y)
            psnr_metric.update(test_pred, y)
        
        
        # Calculate the test loss average per batch
        test_loss /= len(data_loader)
        mse_avg = mse_total / len(data_loader)

        # Calculate the test acc average per batch
        final_ssim = ssim_metric.compute()
        final_psnr = psnr_metric.compute() 
        
        # Return metrics as dictionary
        return {
            'loss': float(test_loss),
            'ssim': float(final_ssim.cpu()),
            'psnr': float(final_psnr.cpu()),
            'mse': float(mse_avg)
        }


def apply_refining_to_dataset(model_instance, refining_model, test_dataset, device, 
                               store_outputs=False, console=None):
    """
    Apply refining to all samples in a dataset and compute metrics.
    
    Args:
        model_instance: The trained model
        refining_model: The refining function from registry
        test_dataset: The test dataset
        device: The device to run on
        store_outputs: Whether to store outputs for visualization
        console: Rich console for progress display (optional)
        
    Returns:
        tuple: (refined_results dict, stored_outputs dict or None)
    """
    refined_results = {}
    stored_outputs = {} if store_outputs else None
    num_samples = len(test_dataset)
    
    if console:
        console.print(f"[yellow]Processing {num_samples} images...[/yellow]")
    
    for i in range(num_samples):
        if console:
            console.print(f"[cyan]Refining image {i+1}/{num_samples}...[/cyan]", end="\r")
        
        fbp_input, original_image = test_dataset[i]
        fbp_input_batch = fbp_input.unsqueeze(0).to(device)
        
        with torch.inference_mode():
            # Get model output
            model_output = model_instance(fbp_input_batch)
            
            # Apply refining
            refined_output_np = refining_model(initial_image=model_output)
            
            # Convert to tensor
            refined_output = torch.from_numpy(refined_output_np).unsqueeze(0).unsqueeze(0).to(device)
            
            # Compute metrics
            mse = torch.nn.functional.mse_loss(refined_output, original_image.unsqueeze(0).to(device)).item()
            psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
            
            # Compute SSIM
            from skimage.metrics import structural_similarity as ssim
            refined_np = refined_output.squeeze().cpu().numpy()
            original_np = original_image.numpy()
            
            # Ensure both arrays are 2D and have the same shape
            if refined_np.ndim != 2:
                refined_np = refined_np.squeeze()
            if original_np.ndim != 2:
                original_np = original_np.squeeze()
            
            # If shapes still don't match, skip SSIM calculation
            if refined_np.shape != original_np.shape:
                ssim_val = 0.0
            else:
                # Ensure images are in the same range
                data_range = original_np.max() - original_np.min()
                if data_range == 0:
                    data_range = 1.0
                
                ssim_val = ssim(refined_np, original_np, data_range=data_range)
            
            refined_results[i] = {
                'mse': mse,
                'psnr': psnr,
                'ssim': ssim_val
            }
            
            # Store outputs if requested
            if store_outputs:
                stored_outputs[i] = {
                    'fbp_input': fbp_input,
                    'original_image': original_image,
                    'model_output': model_output.detach().cpu(),
                    'refined_output': refined_output.detach().cpu(),
                    'sinogram': None
                }
        
        # Clear GPU cache periodically
        if i % 10 == 0:
            torch.cuda.empty_cache()
    
    # Final cache clear
    torch.cuda.empty_cache()
    
    if console:
        console.print("\n")
    
    return refined_results, stored_outputs


def compute_refining_metrics(refined_results, pre_refining_metrics):
    """
    Compute aggregated metrics and improvements from refining results.
    
    Args:
        refined_results: Dictionary with per-sample refined metrics
        pre_refining_metrics: Dictionary with pre-refining metrics (must have 'loss' and 'psnr')
        
    Returns:
        dict: Aggregated metrics with improvements
    """
    avg_refined_mse = np.mean([v['mse'] for v in refined_results.values()])
    avg_refined_psnr = np.mean([v['psnr'] for v in refined_results.values()])
    
    mse_improvement = pre_refining_metrics['loss'] - avg_refined_mse
    psnr_improvement = avg_refined_psnr - pre_refining_metrics['psnr']
    mse_percent = (mse_improvement / pre_refining_metrics['loss'] * 100) if pre_refining_metrics['loss'] > 0 else 0
    
    return {
        'mse': avg_refined_mse,
        'psnr': avg_refined_psnr,
        'mse_improvement': mse_improvement,
        'psnr_improvement': psnr_improvement,
        'mse_improvement_percent': mse_percent
    }


def store_model_outputs(model_instance, test_dataset, device):
    """
    Store model outputs for visualization (without refining).
    
    Args:
        model_instance: The trained model
        test_dataset: The test dataset
        device: The device to run on
        
    Returns:
        dict: Stored outputs for each sample
    """
    stored_outputs = {}
    
    with torch.inference_mode():
        for i in range(len(test_dataset)):
            fbp_input, original_image = test_dataset[i]
            fbp_input_batch = fbp_input.unsqueeze(0).to(device)
            model_output = model_instance(fbp_input_batch)
            
            stored_outputs[i] = {
                'fbp_input': fbp_input,
                'original_image': original_image,
                'model_output': model_output.detach().cpu(),
                'refined_output': None,
                'sinogram': None
            }
            
            # Clear GPU cache periodically
            if i % 10 == 0:
                torch.cuda.empty_cache()
    
    # Final cache clear
    torch.cuda.empty_cache()
    
    return stored_outputs
    
    