import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch
import os
import numpy as np
from PIL import Image, ImageTk
import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
from src.models.esfpnet_module import ESFPModule
import tkinter as tk
import threading

def main(patient_id, durations):
    
    nodule_dir = "/work/hpc/pgl/LIDC-IDRI-Preprocessing/segment_data/Image"
    image_nodule_list = []

    for root, _, files in os.walk(nodule_dir):
        arr = []
        for file in files:
            if file.endswith(".npy"):
                dicom_path = os.path.join(root, file)
                mask_path = dicom_path.replace("Image", "Mask")
                mask_path = mask_path.replace("NI", "MA")
                arr.append(mask_path)
        image_nodule_list.append(arr)
    
    transform = Compose([
        A.Resize(256, 256),
        ToTensorV2(),
    ])
    
    images = []
    i = 0
    
    os.makedirs(f"images/label/LIDC-IDRI-{patient_id}", exist_ok=True)
    
    for file in sorted(image_nodule_list[patient_id]):
        image = np.load(file).astype(np.uint8) * 255
        transformed = transform(image=image)
        image = transformed["image"]
        image = image.squeeze().numpy()
        images.append(Image.fromarray(image))
        
        
        pil_image = Image.fromarray(image) 
        pil_image.save(f"images/label/LIDC-IDRI-{patient_id}/{i}.png")
        i = i + 1
    
    os.makedirs("images/gif/label", exist_ok=True)
    label_gif_path = f"images/gif/label/LIDC-IDRI-{patient_id}.gif"
    images[0].save(label_gif_path, save_all=True, append_images=images[1:], duration=durations, loop=0)

    # Prediction
    prediction_folder = f"/work/hpc/dqm/Lung-Nodule-Segmentation/images/prediction/LIDC-IDRI-{patient_id}"
    image_files = sorted([os.path.join(prediction_folder, file) for file in os.listdir(prediction_folder)], key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    # print(image_files)
    images_pred = [Image.open(path) for path in image_files]
    
    os.makedirs("images/gif/prediction", exist_ok=True)
    
    pred_gif_path = f"images/gif/prediction/LIDC-IDRI-{patient_id}.gif"
    images_pred[0].save(pred_gif_path, save_all=True, append_images=images_pred[1:], duration=durations, loop = 0)  
    
    combine = []
    for i in range(0, len(image_nodule_list[patient_id])):
        if i % 5 == 0:
            mask_array = np.array(images[i])
            colored_mask_array = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)
            colored_mask_array[mask_array == 255] = [0, 255, 0]
            combine.append(Image.fromarray(colored_mask_array))
        else:
            mask_array = np.array(images_pred[i])
            colored_mask_array = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)
            colored_mask_array[mask_array == 255] = [255, 255, 0]
            combine.append(Image.fromarray(colored_mask_array))
            
    os.makedirs("images/gif/combine", exist_ok=True)
    combine_gif_path = f"images/gif/combine/LIDC-IDRI-{patient_id}.gif"

    combine[0].save(combine_gif_path, save_all=True, append_images=combine[1:], duration=durations, loop = 0)

    
if __name__ == "__main__":
    main(313, 500)
    
