import os
import numpy as np
import torch 
import torchvision
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms 

import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2

transformation = Compose(
                [
                    ToTensorV2(),
                ]
            )

def test(image_dir = "/work/hpc/pgl/LIDC-IDRI-Preprocessing/segment_data/Image"):
    file_list = []
        # get full path of each file
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith(".npy"):
                dicom_path = os.path.join(root, file)
                mask_path = dicom_path.replace("Image","Mask")
                mask_path = mask_path.replace("NI","MA")
                if os.path.exists(mask_path):
                    file_list.append((dicom_path, mask_path))
    file_list = np.array(file_list)
    # print(len(file_list)) # 13916
    # print(file_list.shape) # (13916, 2)
    x, y = file_list[0]
    print(x)
    print(y)
    
    image = np.load(x)
    mask = np.load(y)
    image = np.random.randint(5, 250, size=(300, 300), dtype=np.uint8)

    print(image)
    transformed = transformation(image=image)
    image = transformed["image"]
    # mask = transformed["mask"]
    
    # mask = mask.unsqueeze(0).to(torch.float)
    
    # image = torch.from_numpy(image).to(torch.float)
    # mask = torch.from_numpy(mask).to(torch.float)
    
    print(image)
    # print(mask)
    
    
def test_1(image_dir = "/work/hpc/pgl/LIDC-IDRI-Preprocessing/segment_data/Clean/Image"):
    file_list = []
        # get full path of each file
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith(".npy"):
                dicom_path = os.path.join(root, file)
                mask_path = dicom_path.replace("Image","Mask")
                mask_path = mask_path.replace("CN","CM")
                if os.path.exists(mask_path):
                    file_list.append((dicom_path, mask_path))
    file_list = np.array(file_list)
    # print(len(file_list)) # 6885
    # print(file_list.shape) # (6885, 2)
    x, y = file_list[0]
    
    print(x)
    print(y)
    
    image = np.load(x)
    mask = np.load(y)

    
    # print(x1.shape)
    # print(y1.shape)
    
    
    # transformed = transformation(image=image, mask=mask)
    # image = transformed["image"]
    # mask = transformed["mask"] 
    
    image = torch.from_numpy(image).to(torch.float)
    mask = torch.from_numpy(mask).to(torch.float)
    # print(image)
    # print(mask)

if __name__ == "__main__":
    test()
    # test_1()
    