import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.unet import UNet
from datasets.nuclei_dataset import NucleiDataset
from albumentations.pytorch import ToTensorV2
import albumentations as A
import os
from tqdm import tqdm
from nuclear_segmentation import config

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
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5)
        ])
    transforms.extend([
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])
    return A.Compose(transforms)

def main():
    train_dataset = NucleiDataset(
        config.train_image_dir,
        config.train_mask_dir,
        transform=get_transforms()
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    model = UNet(in_channels=config.in_channels, out_channels=config.out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(config.num_epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device).float().unsqueeze(1)
            preds = model(images)
            loss = loss_fn(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), config.checkpoint_path)
    print(f"Model saved to {config.checkpoint_path}")

if __name__ == "__main__":
    main()
