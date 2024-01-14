import wandb

# Initialize the Weights & Biases API
api = wandb.Api()

# Get the run
run_id = "3f9yjeab"
run = api.run(f"minhqd9112003/medical-image/{run_id}")

# Location of file you want to upload
ckpt_path = "/work/hpc/dqm/Lung-Nodule-Segmentation/logs/train/runs/2024-01-10_01-06-52/checkpoints/epoch_134.ckpt"

# Upload
run.upload_file(ckpt_path)
print(f"Successfully upload {ckpt_path}")