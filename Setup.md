## FYP Guide

### Setup Guide
The following steps are copied from the README. Refer to the README for more details if necessary.

1. Clone this repository and navigate to LLaVA folder
```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
```

2. Install Package
```bash
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```bash
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

### Dependency Issues
- During the course of implementation, various dependency issues were faced.
- `llava.yaml` provides a configuration which worked for this project
- Note: cuda-12.4 was used

### Code Intro:
- `code/util.py`: contains utility code
- `code/validate.py`: code to obtain the generated output for evaluation of model performance
- `code/merge_lora_weights.py`: merge lora weights with pretrained weights
