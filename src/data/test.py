import os
import numpy as np

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
    print(len(file_list))
    print(file_list.shape)
    x, y = file_list[0]
    print(x)
    print(y)
    x1 = np.load(x)
    y1 = np.load(y)
    print(x1.shape)
    print(y1.shape)

if __name__ == "__main__":
    test()
    