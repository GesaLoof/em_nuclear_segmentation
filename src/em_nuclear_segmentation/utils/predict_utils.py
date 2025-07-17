# em_nuclear_segmentation/utils/predict_utils.py

import torch
import numpy as np
from PIL import Image
import os
from torchvision.transforms import functional as TF
from skimage.exposure import match_histograms
import matplotlib.pyplot as plt
from em_nuclear_segmentation import config
from models.unet import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def save_visualization(image, mask, filename):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Prediction")
    axes[2].imshow(image, cmap="gray")
    axes[2].imshow(mask, cmap="Reds", alpha=0.4)
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    out_path = os.path.join(config.prediction_output_dir, f"vis_mask_{filename}")
    plt.savefig(out_path)
    plt.close()

def predict(image_path, model=None):
    # Load model if not provided
    if model is None:
        model = UNet(in_channels=config.in_channels, out_channels=config.out_channels)
        model.load_state_dict(torch.load(config.model_path, map_location=device))
        model.to(device).eval()

    # Load image
    original_image = Image.open(image_path)
    image_np = np.array(original_image)

    # Optional: scale uint16 to uint8
    if image_np.dtype == np.uint16:
        image_np = (image_np / 65535.0 * 255).astype(np.uint8)

    # Optional: histogram matching
    if getattr(config, "use_histogram_matching", False):
        ref = Image.open(config.histogram_reference_image)
        ref_np = np.array(ref)
        if ref_np.dtype == np.uint16:
            ref_np = (ref_np / 65535.0 * 255).astype(np.uint8)
        image_np = match_histograms(image_np, ref_np, channel_axis=None)

    image = Image.fromarray(image_np.astype(np.uint8))
    orig_w, orig_h = image.size

    # Preprocess
    image_resized = TF.resize(image, [config.resize_height, config.resize_width])
    tensor = TF.to_tensor(image_resized).float()
    tensor = TF.normalize(tensor, mean=[0.5], std=[0.5])
    tensor = tensor.unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(tensor)
        output = torch.sigmoid(output).squeeze().cpu().numpy()
        pred_mask = (output > 0.5).astype(np.uint8) * 255

    # Resize to original
    pred_mask_resized = Image.fromarray(pred_mask).resize((orig_w, orig_h), resample=Image.NEAREST)

    # Save prediction
    os.makedirs(config.prediction_output_dir, exist_ok=True)
    filename = os.path.basename(image_path)
    mask_path = os.path.join(config.prediction_output_dir, f"mask_{filename}")
    pred_mask_resized.save(mask_path)

    if getattr(config, "save_visual_overlay", False):
        save_visualization(original_image, np.array(pred_mask_resized), filename)

    return pred_mask_resized
