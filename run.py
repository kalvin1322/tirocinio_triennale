"""
Quick launcher for CT Reconstruction Tool
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run
from src.cli.main import app

if __name__ == "__main__":
    print("Starting CT Reconstruction CLI...")
    app()
