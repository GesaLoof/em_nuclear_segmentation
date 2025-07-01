from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np

class NucleiDataset(Dataset):
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
        image = np.array(Image.open(self.image_paths[idx]).convert("L"))
        mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask

