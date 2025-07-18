import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.unet import UNet
from datasets.nuclei_dataset import NucleiDataset
from em_nuclear_segmentation import config
from em_nuclear_segmentation.utils.transform import get_transforms, get_val_transforms
import os
import csv
from tqdm import tqdm

# Device selection
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()

def evaluate(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device).float().unsqueeze(1)
            preds = model(images)
            loss = loss_fn(preds, masks)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def freeze_encoder_layers(model):
    """Freezes the encoder layers of the U-Net model."""
    for layer in model.encoder:
        for param in layer.parameters():
            param.requires_grad = False

def fine_tune():
    # Load datasets
    train_dataset = NucleiDataset(config.train_image_dir, config.train_mask_dir, transform=get_transforms())
    val_dataset = NucleiDataset(config.val_image_dir, config.val_mask_dir, transform=get_val_transforms())
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Initialize model and load pretrained weights
    model = UNet(in_channels=config.in_channels, out_channels=config.out_channels,
    dropout_prob=config.dropout_prob).to(device)
    model.load_state_dict(torch.load(config.fine_tune_from, map_location=device))
    model = model.to(device)

    # Optionally freeze encoder
    if getattr(config, "freeze_encoder", False):
        freeze_encoder_layers(model)
        print("Encoder layers frozen.")

    # Optimizer and loss
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    # Logging setup
    os.makedirs(config.fine_tune_output_dir, exist_ok=True)
    log_path = os.path.join(config.fine_tune_output_dir, "fine_tune_log.csv")
    with open(log_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])

    best_val_loss = float("inf")

    for epoch in range(config.fine_tune_epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Fine-tuning Epoch {epoch+1}/{config.fine_tune_epochs}")

        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device).float().unsqueeze(1)

            preds = model(images)
            loss = loss_fn(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(train_loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, loss_fn)

        # Log per epoch
        with open(log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, val_loss])

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(config.fine_tune_output_dir, f"fine_tuned_{config.fine_tune_output_name}")
            torch.save(model.state_dict(), model_path)
            print(f"Saved fine-tuned model to {model_path} (val_loss = {val_loss:.4f})")

    print("Fine-tuning complete.")

if __name__ == "__main__":
    fine_tune()
