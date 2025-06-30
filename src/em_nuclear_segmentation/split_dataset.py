import os
import shutil
import random
from pathlib import Path
from nuclear_segmentation import config

def split_dataset():
    image_dir = Path(config.raw_image_dir)
    mask_dir = Path(config.raw_mask_dir)
    output_dir = Path(config.split_output_dir)

    assert abs(config.train_split + config.val_split + config.test_split - 1.0) < 1e-6, \
        "Split ratios must sum to 1"

    images = sorted([f for f in image_dir.glob("*") if f.suffix.lower() in [".png", ".jpg", ".tif", ".tiff"]])
    masks = sorted([f for f in mask_dir.glob("*") if f.suffix.lower() in [".png", ".jpg", ".tif", ".tiff"]])
    assert len(images) == len(masks), f"Mismatch: {len(images)} images and {len(masks)} masks"

    paired = list(zip(images, masks))
    random.seed(config.random_seed)
    random.shuffle(paired)

    n_total = len(paired)
    n_train = int(config.train_split * n_total)
    n_val = int(config.val_split * n_total)

    splits = {
        "train": paired[:n_train],
        "val": paired[n_train:n_train + n_val],
        "test": paired[n_train + n_val:]
    }

    for split, files in splits.items():
        for sub in ["images", "masks"]:
            (output_dir / split / sub).mkdir(parents=True, exist_ok=True)
        for img_path, mask_path in files:
            shutil.copy(img_path, output_dir / split / "images" / img_path.name)
            shutil.copy(mask_path, output_dir / split / "masks" / mask_path.name)

    print(f"  Split complete:")
    print(f"  Train: {n_train}")
    print(f"  Val:   {n_val}")
    print(f"  Test:  {n_total - n_train - n_val}")

if __name__ == "__main__":
    split_dataset()
