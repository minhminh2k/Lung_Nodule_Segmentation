from typing import List, Tuple

import os
import hydra
import pyrootutils
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from data.components.LIDC_IDRI_Dataset_predict import LIDC_IDRI_Dataset_Predict


pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from src import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    
    nodule_dir = "/work/hpc/pgl/LIDC-IDRI-Preprocessing/segment_data/Image"
    clean_dir = "/work/hpc/pgl/LIDC-IDRI-Preprocessing/segment_data/Clean/Image"
    
    file_nodule_list = []
    file_clean_list = []

    for root, _, files in os.walk(nodule_dir):
        arr = []
        for file in files:
            if file.endswith(".npy"):
                dicom_path = os.path.join(root, file)
                arr.append(dicom_path)
        file_nodule_list.append(arr)
    # print(file_nodule_list[2])
    
    patient_number = 313
    
    dataset = LIDC_IDRI_Dataset_Predict(file_nodule_list[patient_number], file_clean_list, mode="train", img_size=[256, 256])
    dataloaders = DataLoader(
            dataset=dataset,
            batch_size=8,
            num_workers=0,
            pin_memory=True,
            shuffle=False,
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # for predictions use trainer.predict(...)
    predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)
    # print(type(predictions)) # list
    
    i = 0
    os.makedirs(f"images/prediction/LIDC-IDRI-{patient_number}", exist_ok=True)
    
    for batch in predictions:
        # print(type(batch)) # <class 'torch.Tensor'>
        # print(batch.shape) # torch.Size([4, 1, 256, 256])
        for preds in batch:
            preds = preds.squeeze(0)
            # # preds = F.upsample(preds, size=[256,256], mode='bilinear', align_corners=False)
            preds = preds.sigmoid()
            threshold = torch.tensor([0.5]) # .to(device)
            preds = (preds > threshold).float() * 1
            preds = preds.data.cpu().numpy().squeeze()
            preds = (preds - preds.min()) / (preds.max() - preds.min() + 1e-8)
            preds = torch.from_numpy(preds)
            # print(preds.sum())

            preds_np = preds.numpy()
            preds_np = preds_np * 255
            preds_np = preds_np.astype(np.uint8)
            
            pil_image = Image.fromarray(preds_np) 

            pil_image.save(f"/work/hpc/dqm/Lung-Nodule-Segmentation/images/prediction/LIDC-IDRI-{patient_number}/{i}.png")
            i = i + 1
            
    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
