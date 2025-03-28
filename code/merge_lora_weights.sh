#!/bin/bash

export llava_dir=$HOME/LLaVA

export PYTHONPATH=$llava_dir:$PYTHONPATH

### Fill in the necessary paths to merge the weights
model_name=
model_base=$llava_dir/checkpoints/
lora_adapter=$llava_dir/checkpoints/
save_path=$llava_dir/checkpoints/
vision_tower_path=$llava_dir/checkpoints/
image_processor_path=$llava_dir/checkpoints/

python $llava_dir/scripts/merge_lora_weights.py \
  --model-name $model_name \
  --model-path $lora_adapter \
  --model-base $model_base \
  --save-model-path $save_path \
  --vision_tower_path $vision_tower_path \
  --image_processor_path $image_processor_path
