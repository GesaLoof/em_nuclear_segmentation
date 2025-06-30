import torch
from PIL import Image
from torchvision.transforms import functional as TF
from models.unet import UNet
from nuclear_segmentation import config
import sys
import os
import numpy as np

# Device selection: CUDA > MPS > CPU
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()

def predict(image_path):
    # Load and prepare model
    model = UNet(in_channels=config.in_channels, out_channels=config.out_channels)
    model.load_state_dict(torch.load(config.checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Load and preprocess input image
    image = Image.open(image_path).convert("L")
    image = TF.resize(image, [config.resize_height, config.resize_width])
    tensor = TF.to_tensor(image).unsqueeze(0).to(device)

    # Predict mask
    with torch.no_grad():
        output = model(tensor)
        output = torch.sigmoid(output).squeeze().cpu().numpy()
        mask = (output > 0.5).astype(np.uint8) * 255  # for visualization (0 or 255)

    # Ensure output directory exists
    os.makedirs(config.prediction_output_dir, exist_ok=True)

    # Save output mask
    filename = os.path.basename(image_path)
    out_path = os.path.join(config.prediction_output_dir, f"mask_{filename}")
    Image.fromarray(mask).save(out_path)
    print(f"Prediction saved to {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: predict-nuclei path/to/image.png")
        sys.exit(1)

    predict(sys.argv[1])
