# Data Folder Structure

This folder contains all your CT datasets. Each dataset should follow this structure:

```
data/
├── Mayo_s_Dataset/
│   ├── train/
│   │   ├── C002/
│   │   │   ├── 0.png
│   │   │   ├── 1.png
│   │   │   └── ...
│   │   ├── C004/
│   │   └── ...
│   └── test/
│       ├── C081/
│       │   ├── 0.png
│       │   ├── 1.png
│       │   └── ...
│       └── ...
│
├── Custom_Dataset/
│   ├── train/
│   │   └── ...
│   └── test/
│       └── ...
│
└── Another_Dataset/
    ├── train/
    └── test/
```

## Requirements

Each dataset folder must:
1. Have a `train/` subfolder
2. Have a `test/` subfolder
3. Contain images in subdirectories (e.g., `train/C002/*.png`)

## Adding a New Dataset

1. Create a folder inside `data/` with your dataset name
2. Add `train/` and `test/` subfolders
3. Organize images in subdirectories (one subfolder per patient/scan)
4. Run the wizard - it will automatically detect the new dataset!

## Example

```
data/
└── My_New_Dataset/
    ├── train/
    │   ├── patient_001/
    │   │   ├── slice_0.png
    │   │   ├── slice_1.png
    │   │   └── ...
    │   └── patient_002/
    │       └── ...
    └── test/
        ├── patient_100/
        │   └── ...
        └── patient_101/
            └── ...
```

The wizard will show:
```
✓ Found 1 dataset(s) in data folder:
  • My_New_Dataset - 2000 train, 200 test images
```
