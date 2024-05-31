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
import torchvision
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms 
import tkinter as tk
import threading
import cv2

def normalize_rgb(arr):
    arr_rgb = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8)

    image = Image.fromarray(arr_rgb)

    image = image.convert("RGB")
    return np.array(image)

def mask_overlay(image, mask, color=(0, 1, 0), p = 0.5):
    """Helper function to visualize mask on the top of the image."""
    mask = mask.squeeze() 
    mask = np.dstack((mask, mask, mask)) * np.array(color, dtype=np.uint8)
    weighted_sum = cv2.addWeighted(mask, p, image, p, 0.0)
    img = image.copy()
    ind = mask[:, :, 1] > 0
    img[ind] = weighted_sum[ind]
    return img

def main(patient_id, durations):
    '''
    images_nodule: RGB convert from input image
    images: Label nodule mask
    images_pred: Predict nodule mask
    '''
    # Root
    root_path = "/work/hpc/dqm/Lung-Nodule-Segmentation"
    
    # Read path
    nodule_dir = "/work/hpc/pgl/LIDC-IDRI-Preprocessing/segment_data/Image"
    image_nodule_list = []
    mask_nodule_list = []

    for root, _, files in os.walk(nodule_dir):
        arr_image = []
        arr_mask = []
        for file in files:
            if file.endswith(".npy"):
                dicom_path = os.path.join(root, file)
                arr_image.append(dicom_path)
                mask_path = dicom_path.replace("Image", "Mask")
                mask_path = mask_path.replace("NI", "MA")
                arr_mask.append(mask_path)
        image_nodule_list.append(arr_image)
        mask_nodule_list.append(arr_mask)
        
    
    transform = Compose([
        A.Resize(256, 256),
        ToTensorV2(),
    ])
    
    # Convert numpy nodule image to RGB image
    images_nodule = []
    i = 0
    os.makedirs(f"images/image/LIDC-IDRI-{patient_id}", exist_ok=True)
    
    for file in sorted(image_nodule_list[patient_id]):
        image = np.load(file)
        transformed = transform(image=image)
        image = transformed["image"]        
        image = image.squeeze().numpy()
        image = normalize_rgb(image)
        images_nodule.append(Image.fromarray(image))
    
        nodule = Image.fromarray(image) 
        nodule.save(f"images/image/LIDC-IDRI-{patient_id}/{i}.png")
        i = i + 1
    input_image_list = images_nodule    
    os.makedirs(os.path.join(root_path, "images/gif/input"), exist_ok=True)
    input_image_path_gif = os.path.join(root_path, f"images/gif/input/LIDC-IDRI-{patient_id}.gif")
    input_image_list[0].save(input_image_path_gif, save_all=True, append_images=input_image_list[1:], duration=durations, loop = 0)
    
    # print(image_nodule_list[patient_id])
    # images: label mask
    images = []
    i = 0
    os.makedirs(f"images/label/LIDC-IDRI-{patient_id}", exist_ok=True)
    
    for file in sorted(mask_nodule_list[patient_id]):
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
    
    # images_pred: predicted mask
    images_pred = [Image.open(path) for path in image_files]
    
    os.makedirs("images/gif/prediction", exist_ok=True)
    
    pred_gif_path = f"images/gif/prediction/LIDC-IDRI-{patient_id}.gif"
    images_pred[0].save(pred_gif_path, save_all=True, append_images=images_pred[1:], duration=durations, loop = 0)  
    
    # combine: label + predict mask -> GIF
    combine = []
    for i in range(0, len(mask_nodule_list[patient_id])):
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


    # Overlay Label mask to image: green
    overlay_gif_label = []
    for inp, outp in zip(images_nodule, images):
         overlay_label = mask_overlay(np.array(inp), np.array(outp), color=(0, 1, 0), p = 0.5)
         overlay_label = Image.fromarray(overlay_label)
         overlay_gif_label.append(overlay_label)
    os.makedirs(os.path.join(root_path, "images/gif/overlay/label"), exist_ok=True)
    overlay_gif_label_path = os.path.join(root_path, f"images/gif/overlay/label/LIDC-IDRI-{patient_id}.gif")

    overlay_gif_label[0].save(overlay_gif_label_path, save_all=True, append_images=overlay_gif_label[1:], duration=durations, loop = 0)
    
    # Overlay predict mask to image: yellow
    overlay_gif_predict = []
    for inp, outp in zip(images_nodule, images_pred):
         overlay_predict = mask_overlay(np.array(inp), np.array(outp), color=(1, 1, 0), p = 0.5)
         overlay_predict = Image.fromarray(overlay_predict)
         overlay_gif_predict.append(overlay_predict)
    os.makedirs(os.path.join(root_path, "images/gif/overlay/predict"), exist_ok=True)
    overlay_gif_predict_path = os.path.join(root_path, f"images/gif/overlay/predict/LIDC-IDRI-{patient_id}.gif")
    overlay_gif_predict[0].save(overlay_gif_predict_path, save_all=True, append_images=overlay_gif_predict[1:], duration=durations, loop = 0)
    
    # Overlay label + predict mask to image: green (label) + yellow (predict)
    overlay_gif_label_predict = []
    for inp, lab, pred in zip(images_nodule, images, images_pred):
         overlay_lab = mask_overlay(np.array(inp), np.array(lab), color=(0, 1, 0), p = 0.5)
         overlay_lab_pred = mask_overlay(overlay_lab, np.array(pred), color=(1, 1, 0), p=0.5)
         overlay_lab_pred = Image.fromarray(overlay_lab_pred)
         overlay_gif_label_predict.append(overlay_lab_pred)
    os.makedirs(os.path.join(root_path, "images/gif/overlay/combine"), exist_ok=True)
    overlay_gif_label_predict_path = os.path.join(root_path, f"images/gif/overlay/combine/LIDC-IDRI-{patient_id}.gif")

    overlay_gif_label_predict[0].save(overlay_gif_label_predict_path, save_all=True, append_images=overlay_gif_label_predict[1:], duration=durations, loop = 0)
    
    # Overlay label + predict mask to image : green (label) + yellow (predict): 1 label -> 4 predict -> 1 label -> 4 predict
    overlay_gif_sequence = []
    count = 0
    for inp, lab, pred in zip(images_nodule, images, images_pred):
        if count % 5 == 0:
            overlay_to_input = mask_overlay(np.array(inp), np.array(lab), color=(0, 1, 0), p = 0.5)
            overlay_to_input = Image.fromarray(overlay_to_input)
        else:
            overlay_to_input = mask_overlay(np.array(inp), np.array(pred), color=(1, 1, 0), p = 0.5)
            overlay_to_input = Image.fromarray(overlay_to_input)
            
        overlay_gif_sequence.append(overlay_to_input)
        count = count + 1
    os.makedirs(os.path.join(root_path, "images/gif/overlay/sequence"), exist_ok=True)
    overlay_gif_sequence_path = os.path.join(root_path, f"images/gif/overlay/sequence/LIDC-IDRI-{patient_id}.gif")

    overlay_gif_sequence[0].save(overlay_gif_sequence_path, save_all=True, append_images=overlay_gif_sequence[1:], duration=durations, loop = 0)
    
    
    # Create final result gif
    gif_test_path = os.path.join(root_path, f"images/gif/overlay/label/LIDC-IDRI-{patient_id}.gif")
    gif_test = Image.open(gif_test_path)

    width, height = gif_test.size
   
    frames = []
    for inp, lab, pred, lab_pred, seq in zip(images_nodule, overlay_gif_label, overlay_gif_predict, overlay_gif_label_predict, overlay_gif_sequence):
        combined_image = Image.new('RGB', (5 * width, height))
        combined_image.paste(inp, (0 * width, 0))
        combined_image.paste(lab, (1 * width, 0))
        combined_image.paste(pred, (2 * width, 0))
        combined_image.paste(lab_pred, (3 * width, 0))
        combined_image.paste(seq, (4 * width, 0))
        
        frames.append(combined_image)

    os.makedirs(os.path.join(root_path, "images/gif/result"), exist_ok=True)
    result_gif_path = os.path.join(root_path, f"images/gif/result/LIDC-IDRI-{patient_id}.gif")
    
    frames[0].save(result_gif_path, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    
if __name__ == "__main__":
    main(313, 500)
    
