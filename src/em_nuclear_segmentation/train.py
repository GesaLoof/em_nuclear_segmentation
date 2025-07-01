import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.unet import UNet
from datasets.nuclei_dataset import NucleiDataset
from albumentations.pytorch import ToTensorV2
import albumentations as A
import os
import csv
from tqdm import tqdm
from em_nuclear_segmentation import config

# Device selection: CUDA > MPS > CPU
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()

def get_transforms():
    transforms = [A.Resize(config.resize_height, config.resize_width)]
    if config.use_augmentation:
        transforms.extend([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(translate_percent=0.0625, scale=(0.9, 1.1), rotate=(-15, 15), p=0.5)
        ])
    transforms.extend([
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])
    return A.Compose(transforms)

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

def main():
    # Datasets and loaders
    train_dataset = NucleiDataset(config.train_image_dir, config.train_mask_dir, transform=get_transforms())
    val_dataset = NucleiDataset(config.val_image_dir, config.val_mask_dir, transform=get_transforms())
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Model and optimizer
    model = UNet(in_channels=config.in_channels, out_channels=config.out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    # Loss tracking
    best_val_loss = float("inf")
    log_path = os.path.join(config.split_output_dir, "training_log.csv")
    os.makedirs(config.split_output_dir, exist_ok=True)

    with open(log_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])

        for epoch in range(config.num_epochs):
            model.train()
            running_loss = 0.0
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
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
            print(f"Epoch {epoch+1} complete — Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

            writer.writerow([epoch + 1, avg_train_loss, val_loss])

            # Checkpoint if validation improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(config.split_output_dir, "best_model.pth"))
                print(f"Best model saved (val_loss = {val_loss:.4f})")

    # Save final model
    torch.save(model.state_dict(), config.checkpoint_path)
    print(f"Final model saved to {config.checkpoint_path}")

if __name__ == "__main__":
    main()
