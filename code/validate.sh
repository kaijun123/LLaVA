#!/bin/bash

echo "running"

# edit the path to the llava directory
export llava_dir=$HOME/LLaVA
# echo $llava_dir

export PYTHONPATH=$llava_dir:$PYTHONPATH
# echo $PYTHONPATH

cd $llava_dir
python -u validate.py
echo "end"