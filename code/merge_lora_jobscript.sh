#!/bin/bash

### TC1 Job Script ###
 
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1

### Specify Memory allocate to this job ###
#SBATCH --mem=20G

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

model_name=microsoft-llava-med-v1.5-mistral-7b-train_28k_custom-train-mlp-and-llm-unquantized-epoch-2-lr-6e-5
model_base=/home/FYP/angk0064/ANGK0064/checkpoints/microsoft-llava-med-v1.5-mistral-7b-train_28k_custom-train-mlp-and-llm-unquantized-epoch-2-lr-6e-5
lora_adapter=/home/FYP/angk0064/ANGK0064/checkpoints/lora-microsoft-llava-med-v1.5-mistral-7b-train_28k_custom-train-mlp-and-llm-unquantized-epoch-3-lr-6e-5
save_path=/home/FYP/angk0064/ANGK0064/checkpoints/microsoft-llava-med-v1.5-mistral-7b-train_28k_custom-train-mlp-and-llm-unquantized-epoch-3-lr-6e-5
vision_tower_path=/home/FYP/angk0064/ANGK0064/checkpoints/vision_tower-epoch-1-lr-0.0001
image_processor_path=/home/FYP/angk0064/ANGK0064/checkpoints/vision_tower-epoch-1-lr-0.0001

python $llava_dir/scripts/merge_lora_weights.py \
  --model-name $model_name \
  --model-path $lora_adapter \
  --model-base $model_base \
  --save-model-path $save_path \
  --vision_tower_path $vision_tower_path \
  --image_processor_path $image_processor_path
