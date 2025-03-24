#!/bin/bash

echo "running"
export llava_dir=$HOME/LLaVA
echo $llava_dir
export PYTHONPATH=$llava_dir:$PYTHONPATH
echo $PYTHONPATH
python -u $llava_dir/code/loadModel.py
echo "end"