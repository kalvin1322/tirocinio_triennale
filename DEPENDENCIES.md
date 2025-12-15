# Dependencies and Installation Guide

This document lists all the libraries used in the CT Reconstruction project with manual installation commands.

## Python Version
- **Python 3.12.5** or higher

## PyTorch Installation

The following PyTorch modules are used in this project:
- `torch` (core PyTorch)
- `torchvision` (image transformations and utilities)
- `torchaudio` (audio processing, dependency)
- `torchmetrics` (metrics like SSIM, PSNR)

**Installation depends on your hardware (CPU/CUDA/ROCm).**

Please visit the official PyTorch installation guide and select your configuration:
**https://pytorch.org/get-started/locally/**

Example for CUDA 12.4:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install torchmetrics -c conda-forge
```

---

## Core Scientific Libraries

### NumPy (Numerical Computing)
```bash
conda install numpy=1.26.4
```

### SciPy (Scientific Computing)
```bash
conda install scipy=1.13.1
```

### Scikit-image (Image Processing)
```bash
conda install scikit-image=0.24.0
```

### Scikit-learn (Machine Learning)
```bash
conda install scikit-learn=1.5.1
```

### Matplotlib (Plotting and Visualization)
```bash
conda install matplotlib=3.9.2
```

### Pillow (Image Loading and Manipulation)
```bash
conda install pillow=10.4.0
```

### Pandas (Data Analysis)
```bash
conda install pandas=2.2.2
```

### TQDM (Progress Bars)
```bash
conda install tqdm=4.67.1 -c conda-forge
```

---

## ASTRA Toolbox (CT Reconstruction)

ASTRA Toolbox is critical for CT reconstruction algorithms (FBP, SART, SIRT).

**Installation via Conda (Recommended):**
```bash
conda install -c astra-toolbox astra-toolbox
```

**Note:** ASTRA is not easily installable via pip. Conda is the recommended method.

Official documentation: https://www.astra-toolbox.com/docs/install.html

---

## CLI and UI Libraries

### Rich (Terminal UI with Colors and Tables)
```bash
conda install rich=14.2.0 -c conda-forge
```

### Typer (Modern CLI Framework)
```bash
conda install typer=0.20.0 -c conda-forge
```

### Click (CLI Creation Kit)
```bash
conda install click=8.3.0 -c conda-forge
```

### Inquirer (Interactive Prompts)
```bash
conda install inquirer=3.4.4 -c conda-forge
```

**Inquirer Dependencies:**
```bash
conda install blessed=1.22.0 -c conda-forge
conda install readchar=4.2.1 -c conda-forge
conda install editor=1.6.6 -c conda-forge
```

---

## Configuration and Data Handling

### PyYAML (YAML File Parsing)
```bash
conda install pyyaml=6.0.1
```

---

## Jupyter Notebook Support (Optional)

If you want to use the notebooks in `notebooks/`:

```bash
conda install jupyter=1.0.0
conda install jupyterlab=4.2.5
conda install ipykernel=6.28.0
conda install ipywidgets=8.1.2
conda install notebook=7.2.2
```

---

## Additional Utilities

### FSSpec (Filesystem Spec)
```bash
conda install fsspec=2025.2.0 -c conda-forge
```

### Markdown-it-py (Markdown Parsing)
```bash
conda install markdown-it-py=4.0.0 -c conda-forge
conda install mdurl=0.1.2 -c conda-forge
```

### Lightning Utilities
```bash
conda install lightning-utilities=0.12.0 -c conda-forge
```

---


## Summary of Key Libraries

| Library | Purpose |
|---------|---------|
| PyTorch | Deep learning framework |
| ASTRA Toolbox | CT reconstruction algorithms |
| NumPy | Numerical computing |
| SciPy | Scientific computing |
| Scikit-image | Image processing |
| Scikit-learn | Machine learning utilities |
| Matplotlib | Plotting and visualization |
| Rich | Terminal UI and formatting |
| Typer | CLI framework |
| Click | CLI utilities |
| Inquirer | Interactive prompts |
| PyYAML | Configuration files |
| TorchMetrics | Metrics (PSNR, SSIM) |
| Pillow | Image I/O |
| Pandas | Data handling |
| TQDM | Progress bars |

---

## Notes

1. **All dependencies should be installed via conda** for best compatibility and stability.
2. **ASTRA Toolbox** requires conda installation from the `astra-toolbox` channel and may have CUDA dependencies.
3. **PyTorch** installation varies by platform (CPU/CUDA/ROCm). Always use the official installation command generator.
4. The project was developed with **Python 3.12.5** and **CUDA 12.4**. Other versions may work but are untested.
5. For production use, pin all dependency versions as shown above.
6. The `environment.yml` file provides the complete conda environment specification (recommended method).
7. Many packages require the **conda-forge** channel. Always include `-c conda-forge` when specified.

---

## Troubleshooting

### ASTRA Toolbox Installation Issues
- ASTRA requires CUDA drivers even for CPU usage in some cases
- Use conda installation only
- Check compatibility: https://www.astra-toolbox.com/docs/install.html

### PyTorch CUDA Issues
- Verify CUDA version: `nvidia-smi`
- Match PyTorch CUDA version to your system CUDA version
- Test CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`

### Import Errors
- Ensure virtual environment is activated
- Verify installation: `conda list`
- Check Python version: `python --version`
