from pathlib import Path
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import astra
from torchvision import transforms
from PIL import Image

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent))
from utils.utilities import astra_projection
from utils.geometry_config import load_projection_geometry
from utils.preprocessing_registry import get_preprocessing_method, list_preprocessing_methods


class CTDataset(Dataset):
    def __init__(self, image_path: str, preprocessing_method: str = "FBP", 
                 preprocessing_params: dict = None,
                 geometry_config: str = "default", shape: int = 512):
        """
        CT Dataset with configurable projection geometry and extensible preprocessing methods.
        
        Uses a registry pattern - new preprocessing methods can be added without modifying this class.
        
        Args:
            image_path: Path to image directory
            preprocessing_method: Preprocessing algorithm name (e.g., "FBP", "SART")
            preprocessing_params: Dictionary of parameters for the preprocessing method.
                For FBP: {"filter_type": "ram-lak", "use_cuda": True}
                For SART: {"iterations": 50, "projector_type": "linear"}
            geometry_config: Name of geometry config from JSON file (default: "default")
            shape: Image shape (assumes square images)
        
        Example:
            >>> # FBP with default parameters
            >>> dataset = CTDataset("data/train", preprocessing_method="FBP")
            >>> 
            >>> # SART with custom iterations
            >>> dataset = CTDataset("data/train", preprocessing_method="SART", 
            ...                     preprocessing_params={"iterations": 100})
        """
        super().__init__()
        self.image_paths = sorted(list(Path(image_path).glob("*/*.png")))
        self.preprocessing_method = preprocessing_method.upper()
        self.preprocessing_params = preprocessing_params or {}
        self.geometry_name = geometry_config
        self.vol_geom = astra.create_vol_geom(shape, shape)
        self.proj_geom = load_projection_geometry(geometry_config)
        self.to_tensor = transforms.ToTensor()
        
        # Validate preprocessing method is registered
        self.preprocessing_func = get_preprocessing_method(self.preprocessing_method)
    
    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single dataset item
        
        Returns:
            tuple: (preprocessed_reconstruction_tensor, original_image_tensor)
        """
        # Load original image
        image_path = self.image_paths[idx]
        pil_image = Image.open(image_path).convert("L") 
        original_image_tensor = self.to_tensor(pil_image)
        
        # Create sinogram from original image
        sinogram = astra_projection(original_image_tensor)
        
        # Create ASTRA sinogram data structure
        sinogram_id = astra.data2d.create('-sino', self.proj_geom, data=sinogram)
        
        try:
            # Perform reconstruction using registered preprocessing method
            # The preprocessing_func is retrieved from the registry during __init__
            reconstruction = self.preprocessing_func(
                vol_geom=self.vol_geom,
                sinogram_id=sinogram_id,
                **self.preprocessing_params  # Pass all parameters dynamically
            )

            preprocessed_image_tensor = self.to_tensor(reconstruction.astype(np.float32))
            
        finally:
            # Clean up sinogram data
            astra.data2d.delete(sinogram_id)
        
        return preprocessed_image_tensor, original_image_tensor