from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from em_nuclear_segmentation import config

class NucleiDataset(Dataset):
    """
    Dataset class for loading EM images and their corresponding segmentation masks.

    - Supports optional rescaling of uint16 images to uint8 (via config)
    - Applies Albumentations-style transformations if provided
    - Does NOT convert images to grayscale (no .convert("L"))
    """

    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        assert len(self.image_paths) == len(self.mask_paths), "Image and mask counts must match"
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load raw image
        image = Image.open(self.image_paths[idx])
        mask = Image.open(self.mask_paths[idx])

        # Optional: rescale uint16 images to uint8
        if config.rescale_uint16_to_uint8:
            image_np = np.array(image)
            if image_np.dtype == np.uint16:
                image_np = (image_np / 256).astype(np.uint8)
                image = Image.fromarray(image_np)

        # Apply transform if defined
        if self.transform:
            augmented = self.transform(image=np.array(image), mask=np.array(mask))
            image = augmented['image']
            mask = augmented['mask']

        return image, mask
