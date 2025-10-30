import matplotlib.pyplot as plt
import numpy as np
import astra
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms

def plot_image(image, title="Image"):
    """Display a grayscale image using matplotlib."""
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def astra_projection(image, angles: int = 180, detector_width: int = 768, dist_source_origin: float = 1000, dist_origin_detector: float = 500):
    """
    Create sinogram using ASTRA toolbox with fanbeam geometry
    """
    # Convert PyTorch tensor to numpy array
    if torch.is_tensor(image):
        # If it's a 3D tensor (C, H, W), take the first channel
        if len(image.shape) == 3:
            image_np = image[0].cpu().numpy().astype(np.float32)
        else:
            image_np = image.cpu().numpy().astype(np.float32)
    else:
        image_np = np.array(image, dtype=np.float32)
    
    # Ensure we have a 2D array
    if len(image_np.shape) != 2:
        raise ValueError(f"Expected 2D array, got shape {image_np.shape}")

    image = image_np
    # 1. Creating the volume geometry
    vol_geom = astra.create_vol_geom(image_np.shape[0], image_np.shape[1])

    # 2. Creating the projection geometry
    num_angles = angles       
    angles = np.linspace(0, np.pi, num_angles, False)


    proj_geom = astra.create_proj_geom(
        'fanflat',              # type of geometry
        1.0,                    # distance between the centers of two adjacent detector pixels
        detector_width,         # number of detector pixels
        angles,                 # projection angles in radians
        dist_source_origin,     # distance between the source and the center of rotation DSO
        dist_origin_detector    # distance between the center of rotation and the detector array DOD
    )

    # 5. Creation of the projector
    projector_id = astra.create_projector('cuda', proj_geom, vol_geom)

    # 6. Saving ids of the data
    phantom_id = astra.data2d.create('-vol', vol_geom, data=image_np)
    sinogram_id = astra.data2d.create('-sino', proj_geom)

    # 7. Configura l'algoritmo di Forward Projection (FP)
    cfg = astra.astra_dict('FP_CUDA')
    cfg['ProjectorId'] = projector_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['VolumeDataId'] = phantom_id

    # 8. Creation and execution of the algorithm
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)

    # 9. Retrieve the sinogram calculated from ASTRA's memory
    sinogram = astra.data2d.get(sinogram_id)

    astra.algorithm.delete(alg_id)
    astra.data2d.delete(phantom_id)
    astra.projector.delete(projector_id)
    astra.data2d.delete(sinogram_id)
    return sinogram

def load_image(image_path = "Mayo_s Dataset/train", show= True, random = False, count = 0):
    """
    Load an image from the specified path and apply necessary transformations.
    """
    # 1. get all the images paths (changed from .jpg to .png)
    image_path_list = list(Path(image_path).glob("*/*.png"))
    
    if len(image_path_list) == 0:
        print(f"No images found in {image_path}")
        return None
    
    # 2. get a random image
    if not random:
        random_image_path = image_path_list[count]
    else:
        random_image_path = np.random.choice(image_path_list)

    # 3. taking the patient number from the path
    image_class = random_image_path.parent.stem

    # 4. load and transform the image to tensor
    pil_image = Image.open(random_image_path)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    original_image = transform(pil_image)
    if show:
        plot_image(original_image.permute(1, 2, 0), title=f"Loaded Image - Patient {image_class}")
    sinogram = astra_projection(original_image)
    if show:
        plot_image(sinogram, title="Sinogram")
    return original_image, image_class, sinogram

def print_train_time(start: float,
                     end: float, 
                     device: torch.device = None):
    """Prints difference between start and end time."""
    total_time = end-start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time