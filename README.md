# Stable Diffusion Quantization
This repository demonstrates Quantization-aware Training (QAT) of Stable Diffusion Unet model wich is the most time-consuming element of the whole pipeline. The quantized model is exported to the OpenVINO IR.

The expected speedup from quantization is ~1.7x (for CPUs w/ Intel DL Boost) and can very depeding on the HW.

Knowledge distillation and EMA techniques can be used to improve the model accuracy.

## Prerequisites
```python
pip install -r requirements.txt
```

## HW Requirements
The minimal HW setup for the run is GPU with 24GB of memory.

>**NOTE**: Potentially you can set the number of training steps to 0 and it will lead to Post-Training Quantization. CPU should be enough in this case but you may need to modify the scipt.

## Run QAT
```python
python quantize.py --use_kd --ema_device="cpu" --model_id="runwayml/stable-diffusion-v1-5"
```

`--ema_device="cpu"` is used to save GPU mememory.

