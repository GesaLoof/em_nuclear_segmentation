import torch
from PIL import Image
from torchvision.transforms import functional as TF
from models.unet import UNet
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(image_path):
    model = UNet()
    model.load_state_dict(torch.load("unet_nuclei.pth", map_location=device))
    model.eval()

    image = Image.open(image_path).convert("L")
    image = TF.resize(image, [256, 256])
    tensor = TF.to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        output = torch.sigmoid(output).squeeze().cpu().numpy()
        return output > 0.5

if __name__ == "__main__":
    mask = predict(sys.argv[1])
    print("Prediction complete.")
