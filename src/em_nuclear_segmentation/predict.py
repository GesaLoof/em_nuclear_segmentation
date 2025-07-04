import torch
from PIL import Image
from torchvision.transforms import functional as TF
from models.unet import UNet
from em_nuclear_segmentation import config
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Device selection: CUDA > MPS > CPU
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()

def save_visualization(image, mask, filename):
    """
    Save a side-by-side view of the original image and the predicted mask.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Predicted Mask")
    axes[1].axis("off")

    plt.tight_layout()
    vis_path = os.path.join(config.prediction_output_dir, f"vis_mask_{filename}")
    plt.savefig(vis_path)
    plt.close()
    print(f"Visualization saved to {vis_path}")

def predict(image_path):
    # Load model
    model = UNet(in_channels=config.in_channels, out_channels=config.out_channels)
    model.load_state_dict(torch.load(config.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Load original image and remember original size
    original_image = Image.open(image_path)

    # Convert 16-bit to 8-bit
    if original_image.mode == "I;16" or np.array(original_image).dtype == np.uint16:
        # Convert to numpy array, normalize to 0-1 range, then back to PIL Image
        image_np = np.array(original_image).astype(np.float32)
        image_np /= 65535.0 # Normalize to 0–1 range
        image = Image.fromarray((image_np * 255).astype(np.uint8))  # back to 0–255 range



    orig_w, orig_h = image.size

    # Resize to model input size and normalize to [-1, 1]
    image_resized = TF.resize(image, [config.resize_height, config.resize_width])
    tensor = TF.to_tensor(image_resized).float()
    tensor = TF.normalize(tensor, mean=[0.5], std=[0.5])  # Match training normalization
    tensor = tensor.unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(tensor)
        output = torch.sigmoid(output).squeeze().cpu().numpy()
        print(f"Sigmoid output min/max: {output.min():.4f}, {output.max():.4f}")
        pred_mask = (output > 0.5).astype(np.uint8) * 255

    # Resize predicted mask back to original shape
    pred_mask_resized = Image.fromarray(pred_mask).resize((orig_w, orig_h), resample=Image.NEAREST)

    # Save binary mask
    os.makedirs(config.prediction_output_dir, exist_ok=True)
    filename = os.path.basename(image_path)
    out_path = os.path.join(config.prediction_output_dir, f"mask_{filename}")
    pred_mask_resized.save(out_path)
    print(f"Prediction saved to {out_path}")

    # Optional: Save visualization
    if getattr(config, "save_visual_overlay", False):
        save_visualization(original_image, np.array(pred_mask_resized), filename)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: predict-nuclei path/to/image.tif")
        sys.exit(1)

    predict(sys.argv[1])
