import os
import csv
import numpy as np
import torch
from torchvision.transforms import functional as TF
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.unet import UNet
from datasets.nuclei_dataset import NucleiDataset
from albumentations.pytorch import ToTensorV2
import albumentations as A
from sklearn.metrics import f1_score, accuracy_score, jaccard_score
from em_nuclear_segmentation import config

# Device selection
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()

# Dice score
def dice_coef(y_true, y_pred):
    y_true = y_true.astype(np.bool_)
    y_pred = y_pred.astype(np.bool_)
    intersection = np.logical_and(y_true, y_pred).sum()
    return (2. * intersection) / (y_true.sum() + y_pred.sum() + 1e-8)

# Albumentations test-time transform (resizing, normalization, tensor conversion)
def get_transforms():
    return A.Compose([
        A.Resize(config.resize_height, config.resize_width),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])

# Save comparison grid (original image, mask, prediction, overlay)
def save_visual_comparison(image_np, mask_np, pred_np, filename, metrics):
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    axes[0].imshow(image_np, cmap="gray")
    axes[0].set_title("Input")

    axes[1].imshow(mask_np, cmap="gray")
    axes[1].set_title("Ground Truth")

    axes[2].imshow(pred_np, cmap="gray")
    axes[2].set_title("Prediction")

    axes[3].imshow(image_np, cmap="gray")
    axes[3].imshow(pred_np, cmap="Reds", alpha=0.4)
    axes[3].set_title("Overlay")

    for ax in axes:
        ax.axis("off")

    plt.suptitle(f"F1: {metrics['f1']:.2f} | Dice: {metrics['dice']:.2f} | IoU: {metrics['iou']:.2f} | Acc: {metrics['accuracy']:.2f}")
    plt.tight_layout()
    plt.savefig(f"{config.prediction_output_dir}/vis_{filename}", bbox_inches="tight")
    plt.close()

# Evaluate model on test dataset
def evaluate_model():
    test_dataset = NucleiDataset(config.test_image_dir, config.test_mask_dir, transform=get_transforms())
    model = UNet(in_channels=config.in_channels, out_channels=config.out_channels)
    model.load_state_dict(torch.load(config.evaluation_model_path, map_location=device))
    model.to(device)
    model.eval()

    os.makedirs(config.prediction_output_dir, exist_ok=True)
    metrics = []

    for i in tqdm(range(len(test_dataset)), desc="Evaluating"):
        # Load transformed tensors
        image_tensor, _ = test_dataset[i]
        filename = os.path.basename(test_dataset.image_paths[i])
        image_tensor = image_tensor.unsqueeze(0).to(device)

        # Load original images and masks
        original_image = Image.open(test_dataset.image_paths[i])
        original_mask = Image.open(test_dataset.mask_paths[i])
        orig_w, orig_h = original_image.size

        # Ground truth as binary NumPy array
        mask_np_orig = np.array(original_mask, dtype=np.uint8)

        # Model prediction
        with torch.no_grad():
            output = model(image_tensor)
            pred = torch.sigmoid(output).squeeze().cpu().numpy()
            pred_bin = (pred > 0.5).astype(np.uint8)

        # Resize prediction to original size
        pred_resized = Image.fromarray((pred_bin * 255).astype(np.uint8)).resize((orig_w, orig_h), resample=Image.NEAREST)
        pred_bin_resized = np.array(pred_resized) // 255

        # Flatten for metrics
        y_true = mask_np_orig.flatten()
        y_pred = pred_bin_resized.flatten()

        # Compute metrics
        f1 = f1_score(y_true, y_pred, zero_division=1)
        acc = accuracy_score(y_true, y_pred)
        iou = jaccard_score(y_true, y_pred, zero_division=1)
        dice = dice_coef(mask_np_orig, pred_bin_resized)

        record = {
            "filename": filename,
            "f1": round(f1, 4),
            "accuracy": round(acc, 4),
            "iou": round(iou, 4),
            "dice": round(dice, 4)
        }

        metrics.append(record)

        # Save predicted binary mask (resized)
        if config.save_predictions:
            pred_resized.save(f"{config.prediction_output_dir}/{filename}")

        # Save visualization with original inputs
        if config.save_visualizations:
            save_visual_comparison(
                np.array(original_image),
                mask_np_orig,
                pred_bin_resized,
                filename,
                record
            )

    # Write results to CSV
    with open(config.evaluation_output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)

        # Mean summary
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

