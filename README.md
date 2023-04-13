# Stable Diffusion Quantization
This repository demonstrates Quantization-aware Training (QAT) of Stable Diffusion Unet model wich is the most time-consuming element of the whole pipeline. The quantized model is exported to the OpenVINO IR.

The expected speedup from quantization is ~1.7x (for CPUs w/ Intel DL Boost) and can very depeding on the HW.

Knowledge distillation and EMA techniques can be used to improve the model accuracy.

## Prerequisites
```python
pip install -r requirements.txt
```

Install NNCF from source:
```python
pip install git+https://github.com/openvinotoolkit/nncf.git
```

## HW Requirements
The minimal HW setup for the run is GPU with 24GB of memory.

>**NOTE**: Potentially you can set the number of training steps to 0 and it will lead to Post-Training Quantization. CPU should be enough in this case but you may need to modify the scipt.

## Run PTQ:
```python
python quantize.py --use_kd --center_crop --random_flip --dataset_name="lambdalabs/pokemon-blip-captions" --max_train_steps=0 --model_id="runwayml/stable-diffusion-v1-5"
```

On a part of "laion/laion2B-en" dateset:
```python
python quantize.py --use_kd --center_crop --random_flip --dataset_name="laion/laion2B-en" --model_id="stabilityai/stable-diffusion-2-1" --max_train_samples=500 --dataloader_num_workers=6
```

>**NOTE**: You may need to paly with seed or do one more try to get good results. The results can be better if to use the same dataset that was used to train the original model.

## Run QAT

* Tune all model parameters:
```python
python quantize.py --use_kd --ema_device="cpu" --model_id="runwayml/stable-diffusion-v1-5" --center_crop --random_flip --gradient_checkpointing --scale_lr  --dataloader_num_workers=8 --dataset_name="lambdalabs/pokemon-blip-captions" 
```

`--ema_device="cpu"` and `--gradient_checkpointing` are used to save GPU mememory.

* Tune only quantization parameters. You can use smaller training steps and any relevant dataset:
```python
python quantize.py --use_kd --ema_device="cpu" --model_id="runwayml/stable-diffusion-v1-5" --center_crop --random_flip --gradient_checkpointing --scale_lr --dataloader_num_workers=8 --dataset_name="lambdalabs/pokemon-blip-captions" --tune_quantizers_only 
```


