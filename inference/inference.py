import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch
import os
import numpy as np
from src.models.esfpnet_module import ESFPModule
from PIL import Image
import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
from src.models.esfpnet_module import ESFPModule
import matplotlib.pyplot as plt

def _normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)

    if max_val - min_val > 0:
        image = (image - min_val) / (max_val - min_val)

    return image

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ESFPModule.load_from_checkpoint(
        "checkpoints/epoch_199.ckpt",
        # map_location=torch.device(device),
    )
    
    os.makedirs("images/", exist_ok=True)

    transform = Compose([
        A.Resize(256, 256),
        ToTensorV2(),
    ])
    
    img_path = "/work/hpc/pgl/LIDC-IDRI-Preprocessing/segment_data/Image/LIDC-IDRI-0002/0002_NI000_slice019.npy"
    image = np.load(img_path)
    image = _normalize_image(image)
    transformed = transform(image=image)
    image = transformed["image"]
    image = image.to(torch.float)  # (1, img_size, img_size)
    image = image.unsqueeze(0)

    model.eval()
    
    preds = model(image)
    # print(preds.shape)
    preds = preds.squeeze(0)
    preds = preds.sigmoid()
    threshold = torch.tensor([0.5]) # .to(device)
    preds = (preds > threshold).float() * 1
    preds = preds.data.cpu().numpy().squeeze()
    preds = (preds - preds.min()) / (preds.max() - preds.min() + 1e-8)
    preds = torch.from_numpy(preds)
            
    preds_np = preds.numpy()
    preds_np = preds_np * 255
    preds_np = preds_np.astype(np.uint8)
            
    pil_image = Image.fromarray(preds_np)
    pil_image.save("/work/hpc/dqm/Lung-Nodule-Segmentation/images/one.png")

if __name__ == "__main__":
    main()