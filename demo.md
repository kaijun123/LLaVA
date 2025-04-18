### Steps to launch a demo

#### Dependencies
Ensure that fastapi version is 0.112.2. Use the following command to install if the version is wrong. `pip install fastapi==0.112.2`

#### Launch a controller
```Shell
python -m llava.serve.controller --host 0.0.0.0 --port 10000
```

#### Launch a gradio web server.
```Shell
python -m llava.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload
```
You just launched the Gradio web interface. Now, you can open the web interface with the URL printed on the screen. You may notice that there is no model in the model list. Do not worry, as we have not launched any model worker yet. It will be automatically updated when you launch a model worker.

#### Launch a model worker (LoRA weights, unmerged)

You can launch the model worker with LoRA weights, without merging them with the base checkpoint, to save disk space. There will be additional loading time, while the inference speed is the same as the merged checkpoints. Unmerged LoRA checkpoints do not have `lora-merge` in the model name, and are usually much smaller (less than 1GB) than the merged checkpoints (13G for 7B, and 25G for 13B).

To load unmerged LoRA weights, you simply need to pass an additional argument `--model-base`, which is the base LLM that is used to train the LoRA weights. You can check the base LLM of each LoRA weights in the [model zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md).

```Shell
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 \
--model-path /home/r11kaijun/LLaVA/checkpoints/lora-liuhaotian-llava-v1.5-7b-vision_tower-epoch-1-lr-0.0001-train_28k_custom-train-mlp-and-llm-unquantized-epoch-1-lr-6e-5 \
--model-base /home/r11kaijun/LLaVA/checkpoints/liuhaotian-llava-v1.5-7b \
--image-processor-path /home/r11kaijun/LLaVA/checkpoints/vision_tower-epoch-1-lr-0.0001 \
--vision-tower-path /home/r11kaijun/LLaVA/checkpoints/vision_tower-epoch-1-lr-0.0001
```


