# SVDiff-pytorch
<a href="https://colab.research.google.com/github/mkshing/svdiff-pytorch/blob/main/scripts/svdiff_pytorch.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/svdiff-library/SVDiff-Training-UI)


An implementation of [SVDiff: Compact Parameter Space for Diffusion Fine-Tuning](https://arxiv.org/abs/2303.11305) by using d🧨ffusers. 

My summary tweet is found [here](https://twitter.com/mk1stats/status/1642865505106272257).


![result](assets/dog.png)
left: LoRA, right: SVDiff


Compared with LoRA, the number of trainable parameters is 0.6 M less parameters and the file size is only <1MB (LoRA: 3.1MB)!!

![kumamon](assets/kumamon.png)

## Installation 
```
$ pip install svdiff-pytorch
```
Or, manually 
```bash
$ git clone https://github.com/mkshing/svdiff-pytorch
$ pip install -r requirements.txt
```

## Training
The following example script is for "Single-Subject Generation", which is a domain-tuning on a single object or concept (using 3-5 images). (See Section 4.1)

According to the paper, the learning rate for SVDiff needs to be 1000 times larger than the lr used for fine-tuning. 

```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="path-to-instance-images"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-save-model"

accelerate launch train_svdiff.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="photo of a sks dog" \
  --class_prompt="photo of a dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-3 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800
```


## Inference

```python
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch

from svdiff_pytorch import load_unet_for_svdiff

pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
spectral_shifts_ckpt = "spectral_shifts.safetensors-path"
unet = load_unet_for_svdiff(pretrained_model_name_or_path, spectral_shifts_ckpt=spectral_shifts_ckpt, subfolder="unet")
# load pipe
pipe = StableDiffusionPipeline.from_pretrained(
    pretrained_model_name_or_path,
    unet=unet,
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")
image = pipe("A picture of a sks dog in a bucket", num_inference_steps=25).images[0]
```

You can use the following CLI too! Once it's done, you will see `grid.png` for the result.

```bash
python inference.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"  \
  --spectral_shifts_ckpt="spectral_shifts.safetensors-path"  \
  --prompt="A picture of a sks dog in a bucket"  \
  --scheduler_type="dpm_solver++"  \
  --num_inference_steps=25  \
  --num_images_per_prompt=2 
```

## Additional Features
### Spectral Shift Scaling

![scale](assets/scale.png)

You can adjust the strength of the weights by `--spectral_shifts_scale`

Here's a result for 0.8, 1.0, 1.2 (1.0 is the default).
![scale-result](assets/scale-result.png)


### Fast prior generation by using ToMe
By using [ToMe for SD](https://github.com/dbolya/tomesd), the prior generation can be faster! 
```
$ pip install tomesd
```
And, add `--enable_tome_merging` to your training arguments!

## Citation

```bibtex
@misc{https://doi.org/10.48550/arXiv.2303.11305,
      title         = {SVDiff: Compact Parameter Space for Diffusion Fine-Tuning}, 
      author        = {Ligong Han and Yinxiao Li and Han Zhang and Peyman Milanfar and Dimitris Metaxas and Feng Yang},
      year          = {2023},
      eprint        = {2303.11305},
      archivePrefix = {arXiv},
      primaryClass  = {cs.CV},
      url           = {https://arxiv.org/abs/2303.11305}
}
```

```bibtex
@misc{hu2021lora,
      title         = {LoRA: Low-Rank Adaptation of Large Language Models},
      author        = {Hu, Edward and Shen, Yelong and Wallis, Phil and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Lu and Chen, Weizhu},
      year          = {2021},
      eprint        = {2106.09685},
      archivePrefix = {arXiv},
      primaryClass  = {cs.CL}
}
```

```bibtex
@article{bolya2023tomesd,
      title   = {Token Merging for Fast Stable Diffusion},
      author  = {Bolya, Daniel and Hoffman, Judy},
      journal = {arXiv},
      url     = {https://arxiv.org/abs/2303.17604},
      year    = {2023}
}
```

## Reference
- [DreamBooth in diffusers](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth)
- [DreamBooth in ShivamShrirao](https://github.com/ShivamShrirao/diffusers/tree/main/examples/dreambooth)
- [Data from custom-diffusion](https://github.com/adobe-research/custom-diffusion#getting-started) 

## TODO
- [x] Training
- [x] Inference
- [x] Scaling spectral shifts
- [ ] Support multiple spectral shifts (Section 3.2)
- [ ] Cut-Mix-Unmix (Section 3.3)
- [ ] SVDiff + LoRA
