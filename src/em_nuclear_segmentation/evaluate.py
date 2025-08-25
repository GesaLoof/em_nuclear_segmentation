import os
import csv
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, jaccard_score
from em_nuclear_segmentation import config
from models.unet import UNet
from datasets.nuclei_dataset import NucleiDataset
from em_nuclear_segmentation.utils.predict_utils import predict  # uses shared prediction logic

# Dice coefficient
def dice_coef(y_true, y_pred):
    y_true = y_true.astype(np.bool_)
    y_pred = y_pred.astype(np.bool_)
    intersection = np.logical_and(y_true, y_pred).sum()
    return (2. * intersection) / (y_true.sum() + y_pred.sum() + 1e-8)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS works on Apple Silicon/macOS with PyTorch 1.12+
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# Main evaluation loop
def evaluate_model():
    # Load test dataset (no transform required; handled inside predict function)
    test_dataset = NucleiDataset(config.test_image_dir, config.test_mask_dir, transform=None)
    device = get_device()
    model = UNet(in_channels=config.in_channels, out_channels=config.out_channels).to(device)
    # Load weights onto the same device as the model
    state = torch.load(config.evaluation_model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    # Ensure output directory exists

    os.makedirs(config.prediction_output_dir, exist_ok=True)
    metrics = []

    for i in tqdm(range(len(test_dataset)), desc="Evaluating"):
        image_path = test_dataset.image_paths[i]
        mask_path = test_dataset.mask_paths[i]
        filename = os.path.basename(image_path)

        # Run prediction using shared logic
        pred_mask = np.array(predict(image_path, model=model))

        # Load ground truth
        gt_mask = np.array(Image.open(mask_path), dtype=np.uint8)

        # Binarize predicted mask if needed
        if pred_mask.max() > 1:
            pred_mask = pred_mask // 255

        # Flatten masks for metric computation
        y_true = gt_mask.flatten()
        y_pred = pred_mask.flatten()

        # Compute metrics
        f1 = f1_score(y_true, y_pred, zero_division=1)
        acc = accuracy_score(y_true, y_pred)
        iou = jaccard_score(y_true, y_pred, zero_division=1)
        dice = dice_coef(y_true, y_pred)

        record = {
            "filename": filename,
            "f1": round(f1, 4),
            "accuracy": round(acc, 4),
            "iou": round(iou, 4),
            "dice": round(dice, 4)
        }

        metrics.append(record)

    # Write all results to CSV
    with open(config.evaluation_output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)

        # Write mean row
        mean = {
            "filename": "MEAN",
            "f1": round(np.mean([m["f1"] for m in metrics]), 4),
            "accuracy": round(np.mean([m["accuracy"] for m in metrics]), 4),
            "iou": round(np.mean([m["iou"] for m in metrics]), 4),
            "dice": round(np.mean([m["dice"] for m in metrics]), 4),
        }
        writer.writerow(mean)

    print(f"Evaluation complete. Results saved to {config.evaluation_output_csv}")

if __name__ == "__main__":
    evaluate_model()
