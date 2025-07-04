import os
import csv
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
import argparse
import matplotlib.pyplot as plt
from em_nuclear_segmentation import config

def crop_pair(image, mask, patch_size, edge_mode):
    """
    Crops an image-mask pair into patches according to edge handling strategy.

    Parameters:
        image (PIL.Image): The input grayscale image.
        mask (PIL.Image): The corresponding binary mask.
        patch_size (int): The size (width and height) of each patch.
        edge_mode (str): One of "pad", "keep", or "overlap".

    Yields:
        Tuple[PIL.Image, PIL.Image, Tuple[int, int]]:
            - image_patch: The cropped image patch
            - mask_patch: The cropped mask patch
            - (x, y): Top-left coordinates of the patch in the original image
    """
    w, h = image.size

    # If the image is already smaller than patch size, return it as is
    if w <= patch_size and h <= patch_size:
        yield image, mask, (0, 0)
        return

    # Handle edge padding
    if edge_mode == "pad":
        pad_w = patch_size - (w % patch_size) if w % patch_size != 0 else 0
        pad_h = patch_size - (h % patch_size) if h % patch_size != 0 else 0

        # Determine fill value for image padding based on config
        if config.pad_fill_mode == "zero":
            fill_val = 0
        elif config.pad_fill_mode == "mean":
            fill_val = int(np.array(image).mean())
        elif config.pad_fill_mode == "median":
            fill_val = int(np.median(np.array(image)))
        elif config.pad_fill_mode.startswith("value:"):
            try:
                fill_val = int(config.pad_fill_mode.split(":")[1])
            except:
                raise ValueError(f"Invalid pad_fill_mode: {config.pad_fill_mode}")
        else:
            raise ValueError(f"Unsupported pad_fill_mode: {config.pad_fill_mode}")

        # Apply padding to both image and mask
        image = ImageOps.expand(image, (0, 0, pad_w, pad_h), fill=fill_val)
        mask = ImageOps.expand(mask, (0, 0, pad_w, pad_h), fill=0)
        w, h = image.size

    step = patch_size

    if edge_mode == "overlap":
        # Slide window with overlap at edges to fully cover image
        for top in range(0, h, patch_size):
            for left in range(0, w, patch_size):
                if top + patch_size > h:
                    top = h - patch_size
                if left + patch_size > w:
                    left = w - patch_size
                box = (left, top, left + patch_size, top + patch_size)
                yield image.crop(box), mask.crop(box), (left, top)
    else:
        # Standard non-overlapping tiling
        for top in range(0, h, step):
            for left in range(0, w, step):
                box = (left, top, left + patch_size, top + patch_size)
                yield image.crop(box), mask.crop(box), (left, top)

def plot_size_distribution(original_sizes, patch_sizes, output_path):
    """
    Plots histograms of original image sizes and patch sizes (in pixels).

    Parameters:
        original_sizes (List[Tuple[int, int]]): Sizes of original images.
        patch_sizes (List[Tuple[int, int]]): Sizes of generated patches.
        output_path (str or Path): Where to save the output plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist([w * h for w, h in original_sizes], bins=20, color='skyblue', edgecolor='black')
    axes[0].set_title("Original Image Size Distribution")
    axes[0].set_xlabel("Pixels (W × H)")
    axes[0].set_ylabel("Count")

    axes[1].hist([w * h for w, h in patch_sizes], bins=20, color='salmon', edgecolor='black')
    axes[1].set_title("Patch Size Distribution")
    axes[1].set_xlabel("Pixels (W × H)")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved size distribution plot to {output_path}")

def generate_preview_grid(patch_dir, output_path, n=25):
    """
    Generates a preview grid image of the first N patches.

    Parameters:
        patch_dir (Path): Directory containing patch images.
        output_path (Path): Output path for the preview image.
        n (int): Number of patches to display (must form a square grid).
    """
    patch_files = sorted(Path(patch_dir).glob("*.png"))[:n]
    if not patch_files:
        print("No patches found to generate preview.")
        return

    patch_images = [Image.open(p) for p in patch_files]
    patch_size = patch_images[0].size
    grid_size = int(n ** 0.5)
    grid_img = Image.new("L", (patch_size[0] * grid_size, patch_size[1] * grid_size))

    for idx, patch in enumerate(patch_images):
        x = (idx % grid_size) * patch_size[0]
        y = (idx // grid_size) * patch_size[1]
        grid_img.paste(patch, (x, y))

    grid_img.save(output_path)
    print(f"Saved preview grid to {output_path}")

def process_images_and_masks():
    """
    Processes a folder of raw images and masks into patches.
    Saves:
        - Cropped image and mask patches
        - Patch metadata CSV
        - Size distribution plot
        - Preview grid of patches
    """
    # Define directories
    image_dir = Path(config.raw_image_dir)
    mask_dir = Path(config.raw_mask_dir)
    output_image_dir = Path(config.cropped_output_dir) / "images"
    output_mask_dir = Path(config.cropped_output_dir) / "masks"
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = Path(config.cropped_output_dir) / "patch_metadata.csv"
    preview_path = Path(config.preview_grid_path)
    metadata = []

    # Load and sort input files
    image_files = sorted([f for f in image_dir.glob("*") if f.suffix.lower() in [".png", ".jpg", ".tif", ".tiff"]])
    mask_files = sorted([f for f in mask_dir.glob("*") if f.suffix.lower() in [".png", ".jpg", ".tif", ".tiff"]])
    assert len(image_files) == len(mask_files), "Mismatch in number of images and masks"

    original_sizes = []
    patch_sizes = []
    patch_index = 1

    # Process each image-mask pair
    for img_path, mask_path in zip(image_files, mask_files):
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")
        w, h = image.size
        original_sizes.append((w, h))
        stem = img_path.stem

        for img_patch, mask_patch, (x, y) in crop_pair(image, mask, config.patch_size, config.edge_mode):
            patch_sizes.append(img_patch.size)
            filename = f"image{patch_index}.png"

            img_patch.save(output_image_dir / filename)
            mask_patch.save(output_mask_dir / filename)

            metadata.append({
                "patch_filename": filename,
                "source_image": img_path.name,
                "origin_x": x,
                "origin_y": y,
                "width": img_patch.width,
                "height": img_patch.height
            })

            patch_index += 1

        print(f"Processed {img_path.name} and {mask_path.name}")

    # Save patch metadata
    with open(metadata_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metadata[0].keys())
        writer.writeheader()
        writer.writerows(metadata)
    print(f"Saved patch metadata to {metadata_path}")

    # Save plots
    plot_size_distribution(original_sizes, patch_sizes, config.plot_path)
    generate_preview_grid(output_image_dir, preview_path, n=config.max_preview_patches)

if __name__ == "__main__":
    process_images_and_masks()
