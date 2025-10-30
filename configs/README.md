# Projection Geometry Configuration - Quick Guide

## 📝 Summary

The projection geometry system is designed to be **simple and user-friendly**. Users configure geometries by editing a JSON file - no Python code required.

## 🎯 How It Works

### 1. Edit the JSON File

Open `configs/projection_geometry.json` and add/modify geometries:

```json
{
  "my_experiment": {
    "geometry_type": "fanflat",
    "detector_spacing": 1.0,
    "detector_count": 768,
    "num_angles": 180,
    "angle_range": [0, 3.14159265359],
    "source_origin_distance": 1000.0,
    "origin_detector_distance": 500.0,
    "description": "My custom configuration"
  }
}
```

### 2. Use It in Your Dataset

```python
from src.dataloader.CTDataloader import CTDataset

# Use your custom geometry
dataset = CTDataset(
    image_path="Mayo_s Dataset/train",
    geometry_config="my_experiment"  # ← Just use the name from JSON
)
```

That's it! No need to write low-level ASTRA code.

## 📋 Pre-configured Geometries

The config file includes 5 ready-to-use geometries:

| Name | Description | Use Case |
|------|-------------|----------|
| `default` | Standard fanbeam | General training/testing |
| `high_resolution` | More angles & pixels | Final benchmarking |
| `low_dose` | Fewer angles | Quick prototyping |
| `parallel_beam` | Parallel geometry | Educational/simple cases |
| `cone_beam` | Full 360° rotation | Advanced experiments |

## 🔧 Required Parameters

All geometries must include:

- `geometry_type`: `"parallel"` or `"fanflat"`
- `detector_spacing`: Usually `1.0`
- `detector_count`: Number of detector pixels
- `num_angles`: Number of projection angles
- `angle_range`: `[start, end]` in radians

For fanbeam geometries, also include:
- `source_origin_distance`: DSO distance
- `origin_detector_distance`: DOD distance

## 💡 Tips

✅ **Copy existing geometries** as templates  
✅ **Use descriptive names** for your experiments  
✅ **Add descriptions** to remember what each config does  
✅ **Validate JSON** syntax before running (use jsonlint.com)  

## ⚠️ Common Mistakes

❌ Forgetting required parameters  
❌ Invalid JSON syntax (missing commas, quotes)  
❌ Wrong geometry type name  
❌ Angle range not in radians  

## 📖 Full Documentation

See `docs/projection_geometry_guide.md` for complete parameter descriptions and examples.
