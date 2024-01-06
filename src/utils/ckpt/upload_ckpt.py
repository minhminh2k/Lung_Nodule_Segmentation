import wandb

# Initialize the Weights & Biases API
api = wandb.Api()

# Get the run
run_id = "ictom8y2"
run = api.run(f"minhqd9112003/medical-image/{run_id}")

# Location of file you want to upload
ckpt_path = "/work/hpc/dqm/Lung-Nodule-Segmentation/logs/train/runs/2024-01-04_13-38-03/checkpoints/epoch_064.ckpt"

# Upload
run.upload_file(ckpt_path)
print(f"Successfully upload {ckpt_path}")