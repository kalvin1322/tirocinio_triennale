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
from models.FBP_recostruction import run_fbp_reconstruction


class CTDataset(Dataset):
    def __init__(self, image_path: str, filter_type: str = "ram-lak", 
                 geometry_config: str = "default", shape: int = 512, use_cuda: bool = True):
        """
        CT Dataset with configurable projection geometry
        
        Args:
            image_path: Path to image directory
            filter_type: FBP filter type (ram-lak, shepp-logan, cosine, etc.)
            geometry_config: Name of geometry config from JSON file (default: "default")
            shape: Image shape (assumes square images)
            use_cuda: Whether to use CUDA for FBP reconstruction
        """
        super().__init__()
        self.image_paths = sorted(list(Path(image_path).glob("*/*.png")))
        self.filter_type = filter_type
        self.use_cuda = use_cuda
        self.geometry_name = geometry_config
        self.vol_geom = astra.create_vol_geom(shape, shape)
        self.proj_geom = load_projection_geometry(geometry_config)
        self.to_tensor = transforms.ToTensor()
    
    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single dataset item
        
        Returns:
            tuple: (fbp_reconstruction_tensor, original_image_tensor)
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
            # Perform FBP reconstruction using the dedicated function
            reconstruction = run_fbp_reconstruction(
                vol_geom=self.vol_geom,
                sinogram_id=sinogram_id,
                filter_type=self.filter_type,
                use_cuda=self.use_cuda
            )

            fbp_image_tensor = self.to_tensor(reconstruction.astype(np.float32))
            
        finally:
            # Clean up sinogram data
            astra.data2d.delete(sinogram_id)
        
        return fbp_image_tensor, original_image_tensor