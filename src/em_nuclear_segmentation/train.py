import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.unet import UNet
from datasets.nuclei_dataset import NucleiDataset
import os
import csv
from tqdm import tqdm
from em_nuclear_segmentation import config
from em_nuclear_segmentation.utils.transform import get_transforms, get_val_transforms

# Device selection
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()

# Validation evaluation
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
    # Load datasets
    train_dataset = NucleiDataset(config.train_image_dir, config.train_mask_dir, transform=get_transforms())
    val_dataset = NucleiDataset(config.val_image_dir, config.val_mask_dir, transform=get_val_transforms())
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Initialize model
    model = UNet(in_channels=config.in_channels, out_channels=config.out_channels, dropout_prob = config.dropout_prob).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    os.makedirs(config.train_output_dir, exist_ok=True)
    log_path = os.path.join(config.train_output_dir, "training_log.csv")
    best_model_path = os.path.join(config.train_output_dir, f"best_{config.model_output_name}")
    final_model_path = os.path.join(config.train_output_dir, f"final_{config.model_output_name}")

    start_epoch = 0
    best_val_loss = float("inf")

    # Resume support
    if not os.path.exists(config.resume_checkpoint_path):
        print("No checkpoint found. Starting fresh training.")

    if getattr(config, "resume_training", False) and os.path.exists(config.resume_checkpoint_path):
        print(f"Resuming from checkpoint: {config.resume_checkpoint_path}")
        model.load_state_dict(torch.load(config.resume_checkpoint_path, map_location=device))

        if os.path.exists(log_path):
            with open(log_path, newline="") as f:
                reader = csv.DictReader(f)
                history = list(reader)
            if history:
                last_entry = history[-1]
                start_epoch = int(last_entry["epoch"])
                best_val_loss = float(last_entry["val_loss"])
                print(f"Resuming at epoch {start_epoch+1} with best_val_loss = {best_val_loss:.4f}")
        else:
            print("No training log found — starting from checkpoint but logging fresh.")
            with open(log_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "val_loss"])
    else:
        # Start fresh log
        with open(log_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss"])

    patience_counter = 0

    for epoch in range(start_epoch, config.num_epochs):
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

        # Log results
        with open(log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, val_loss])

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to {best_model_path} (val_loss = {val_loss:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if config.use_early_stopping and patience_counter >= config.early_stopping_patience:
            print(f"Early stopping triggered after {patience_counter} epochs without improvement.")
            break

    # Save final model
    torch.save(model.state_dict(), final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")

if __name__ == "__main__":
    main()
