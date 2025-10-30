# Projection Geometry Configuration Guide

This guide explains how to configure and use custom projection geometries for CT reconstruction experiments.

## 📁 Configuration File

The projection geometries are defined in `configs/projection_geometry.json`. Each geometry configuration includes parameters for the ASTRA toolbox projection setup.

## 🔧 Available Geometries

### 1. **default** (Recommended)
Standard fanbeam geometry for general CT reconstruction:
- 768 detector pixels
- 180 projection angles
- Fanbeam with DSO=1000, DOD=500

### 2. **high_resolution**
Higher quality reconstruction with more data:
- 1024 detector pixels
- 360 projection angles
- Better quality but slower

### 3. **low_dose**
Simulates low-dose CT with fewer projections:
- 512 detector pixels
- 90 projection angles
- Faster but lower quality

### 4. **parallel_beam**
Simple parallel beam geometry:
- Good for educational purposes
- Simpler reconstruction algorithms

### 5. **cone_beam**
Full 360° rotation cone beam:
- 896 detector pixels
- 200 angles with full rotation

## 💻 Usage Examples

### Basic Usage in Python

```python
from src.dataloader.CTDataloader import CTDataset
from torch.utils.data import DataLoader

# Use default geometry
train_dataset = CTDataset(
    image_path="Mayo_s Dataset/train",
    geometry_config="default"
)

# Use high resolution geometry
train_dataset_hr = CTDataset(
    image_path="Mayo_s Dataset/train",
    geometry_config="high_resolution"
)

# Use low dose geometry
train_dataset_ld = CTDataset(
    image_path="Mayo_s Dataset/train",
    geometry_config="low_dose"
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
```

### Load Geometry in Code

```python
from src.utils.geometry_config import load_projection_geometry

# Load a geometry by name
proj_geom = load_projection_geometry("default")
proj_geom_hr = load_projection_geometry("high_resolution")
```

### View Available Geometries

To see all available geometries, simply open the JSON file:

```bash
# View the configuration file
cat configs/projection_geometry.json

# Or open in your editor
code configs/projection_geometry.json
```

## ✏️ Creating Custom Geometries

Simply edit the `configs/projection_geometry.json` file and add your custom geometry:

```json
{
  "my_custom_geometry": {
    "geometry_type": "fanflat",
    "detector_spacing": 1.0,
    "detector_count": 600,
    "num_angles": 120,
    "angle_range": [0, 3.14159265359],
    "source_origin_distance": 800.0,
    "origin_detector_distance": 400.0,
    "description": "My custom geometry for specific experiment"
  }
}
```

Then use it directly:

```python
from src.dataloader.CTDataloader import CTDataset

dataset = CTDataset(
    image_path="Mayo_s Dataset/train",
    geometry_config="my_custom_geometry"  # ← Use your new geometry
)
```

## 📊 Parameter Descriptions

### Required Parameters (All Geometries)

- **geometry_type**: Type of projection geometry
  - `"parallel"`: Parallel beam
  - `"fanflat"` or `"fanbeam"`: Fanbeam geometry

- **detector_spacing**: Distance between detector pixels (usually 1.0)

- **detector_count**: Number of detector pixels
  - Higher = better resolution but slower
  - Typical range: 512-1024

- **num_angles**: Number of projection angles
  - Higher = better quality but slower
  - Typical range: 90-360

- **angle_range**: `[start, end]` in radians
  - Half scan: `[0, π]` (3.14159...)
  - Full scan: `[0, 2π]` (6.28318...)

### Fanbeam-Specific Parameters

- **source_origin_distance** (DSO): Distance from X-ray source to rotation center
  - Typical: 1000 units

- **origin_detector_distance** (DOD): Distance from rotation center to detector
  - Typical: 500 units

- **Magnification**: `(DSO + DOD) / DSO`

### Optional Parameters

- **description**: Human-readable description of the geometry



## 🔍 Troubleshooting

### Error: "Geometry not found"
- Check the name matches exactly what's in the JSON file
- Use `config.list_geometries()` to see available names

### Error: "Missing required parameter"
- Ensure all required parameters are present in your custom geometry
- Check JSON syntax is valid

### Poor reconstruction quality
- Try increasing `num_angles`
- Try increasing `detector_count`
- Check that angle_range is appropriate for your application

### Slow performance
- Reduce `num_angles` (e.g., 180 → 90)
- Reduce `detector_count` (e.g., 768 → 512)
- Use `low_dose` geometry for prototyping

## 📚 Additional Resources

- ASTRA Toolbox documentation: https://www.astra-toolbox.com/
