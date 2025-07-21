#!/bin/bash
#SBATCH --job-name=smolvla_training             # Name of your job
#SBATCH --output=%x_%j.out            # Output file (%x for job name, %j for job ID)
#SBATCH --error=%x_%j.err             # Error file
#SBATCH --partition=A100              # Partition to submit to (A100, V100, etc.)
#SBATCH --gres=gpu:2                  # Request 1 GPU
#SBATCH --mem=32G                     # Request 32 GB of memory
#SBATCH --time=24:00:00               # Time limit for the job (hh:mm:ss)

# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# # Define variables for your job
# DATA_DIR="~/data"
# LR="1e-3"
# EPOCHS=100
# BATCH_SIZE=32

# Activate the environment
# source ~/miniconda3/condabin/conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate smolvla
export PYTHONPATH=$PYTHONPATH:/home/ids/ext-5219/Benchmark_VLAs
# Execute the Python script with specific arguments
srun python ~/Benchmark_VLAs/lerobot/src/lerobot/scripts/train.py \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=lerobot/svla_so101_pickplace \
  --batch_size=64 \
  --steps=20000 \
  --output_dir=outputs/train/my_smolvla \
  --job_name=my_smolvla_training \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.push_to_hub=false


# Print job completion time
echo "Job finished at: $(date)"