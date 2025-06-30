import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.unet import UNet
from datasets.nuclei_dataset import NucleiDataset
from albumentations.pytorch import ToTensorV2
import albumentations as A
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_transforms():
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])

def main():
    train_dataset = NucleiDataset("data/train/images", "data/train/masks", transform=get_transforms())
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(10):
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

    torch.save(model.state_dict(), "unet_nuclei.pth")

if __name__ == "__main__":
    main()
