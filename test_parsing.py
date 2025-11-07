"""
Test script to debug checkpoint filename parsing
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils.model_params import parse_model_params_from_filename

# Test cases
test_cases = [
    ("FBP_PostProcessNet_hc16_ep50_lr0001.pth", "PostProcessNet"),
    ("FBP_UNet_V1_enc4_ch128_ep50_lr0001.pth", "UNet_V1"),
    ("SIRT_SimpleResNet_lay5_fea64_ep100_lr00001.pth", "SimpleResNet"),
    ("FBP_ThreeL_SSNet_ep50_lr0001.pth", "ThreeL_SSNet"),
]

print("Testing checkpoint filename parsing:\n")
for filename, model in test_cases:
    print(f"File: {filename}")
    print(f"Model: {model}")
    try:
        params = parse_model_params_from_filename(filename, model)
        print(f"Parsed params: {params}")
    except Exception as e:
        print(f"ERROR: {e}")
    print("-" * 60)
