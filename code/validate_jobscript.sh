#!/bin/bash

### TC1 Job Script ###
 
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1

### Specify Memory allocate to this job ###
#SBATCH --mem=10G

### Specify number of core (CPU) to allocate to per node ###
#SBATCH --ntasks-per-node=1

### Specify number of node to compute ###
#SBATCH --nodes=1

### Optional: Specify node to execute the job ###
### Remove 1st # at next line for the option to take effect ###
##SBATCH --nodelist=TC1N07

### Specify Time Limit, format: <min> or <min>:<sec> or <hr>:<min>:<sec> or <days>-<hr>:<min>:<sec> or <days>-<hr> ### 
#SBATCH --time=360

### Specify name for the job, filename format for output and error ###
#SBATCH --job-name=TestJob
SBATCH --output=output_%x_%j.out
SBATCH --error=error_%x_%j.err

### Your script for computation ###
module load anaconda
source activate llava

# edit the path to the llava directory
export llava_dir=$HOME/LLaVA
export PYTHONPATH=$llava_dir:$PYTHONPATH

python -u validate.py \
  --model_path /home/FYP/angk0064/ANGK0064/checkpoints/lora-microsoft-llava-med-v1.5-mistral-7b-train_28k_custom-train-mlp-and-llm-unquantized-epoch-3-lr-6e-5 \
  --model_base /home/FYP/angk0064/ANGK0064/checkpoints/microsoft-llava-med-v1.5-mistral-7b-train_28k_custom-train-mlp-and-llm-unquantized-epoch-2-lr-6e-5 \
  --model_name lora-microsoft-llava-med-v1.5-mistral-7b-train_28k_custom-train-mlp-and-llm-unquantized-epoch-3-lr-6e-5 \
  --image_processor_path /home/FYP/angk0064/ANGK0064/checkpoints/vision_tower-epoch-1-lr-0.0001 \
  --vision_tower_path /home/FYP/angk0064/ANGK0064/checkpoints/vision_tower-epoch-1-lr-0.0001 \
  --image-base-path /home/FYP/angk0064/Datasets/mimic-cxr-jpg/2.1.0 \
  --question-file /home/FYP/angk0064/Datasets/mimic-cxr/processed_data/validate_custom.json \
  --answers-file lora-microsoft-llava-med-v1.5-mistral-7b-train_28k_custom-train-mlp-and-llm-unquantized-epoch-3-lr-6e-5.json