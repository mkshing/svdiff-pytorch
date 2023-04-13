import os
import inspect
from PIL import Image

import torch
import accelerate
from accelerate.utils import set_module_tensor_to_device
from diffusers import (
    LMSDiscreteScheduler,
    DDIMScheduler,
    PNDMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
)
from transformers import CLIPTextModel, CLIPTextConfig
from diffusers import UNet2DConditionModel
from safetensors.torch import safe_open
import huggingface_hub
from svdiff_pytorch import UNet2DConditionModelForSVDiff, CLIPTextModelForSVDiff



def load_unet_for_svdiff(pretrained_model_name_or_path, spectral_shifts_ckpt=None, hf_hub_kwargs=None, **kwargs):
    """
    https://github.com/huggingface/diffusers/blob/v0.14.0/src/diffusers/models/modeling_utils.py#L541
    """
    config = UNet2DConditionModel.load_config(pretrained_model_name_or_path, **kwargs)
    original_model = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
    state_dict = original_model.state_dict()
    with accelerate.init_empty_weights():
        model = UNet2DConditionModelForSVDiff.from_config(config)
    # load pre-trained weights
    param_device = "cpu"
    torch_dtype = kwargs["torch_dtype"] if "torch_dtype" in kwargs else None
    spectral_shifts_weights = {n: torch.zeros(p.shape) for n, p in model.named_parameters() if "delta" in n}
    state_dict.update(spectral_shifts_weights)
    # move the params from meta device to cpu
    missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
    if len(missing_keys) > 0:
        raise ValueError(
            f"Cannot load {model.__class__.__name__} from {pretrained_model_name_or_path} because the following keys are"
            f" missing: \n {', '.join(missing_keys)}. \n Please make sure to pass"
            " `low_cpu_mem_usage=False` and `device_map=None` if you want to randomely initialize"
            " those weights or else make sure your checkpoint file is correct."
        )

    for param_name, param in state_dict.items():
        accepts_dtype = "dtype" in set(inspect.signature(set_module_tensor_to_device).parameters.keys())
        if accepts_dtype:
            set_module_tensor_to_device(model, param_name, param_device, value=param, dtype=torch_dtype)
        else:
            set_module_tensor_to_device(model, param_name, param_device, value=param)
    
    if spectral_shifts_ckpt:
        if os.path.isdir(spectral_shifts_ckpt):
            spectral_shifts_ckpt = os.path.join(spectral_shifts_ckpt, "spectral_shifts.safetensors")
        elif not os.path.exists(spectral_shifts_ckpt):
            # download from hub
            hf_hub_kwargs = {} if hf_hub_kwargs is None else hf_hub_kwargs
            spectral_shifts_ckpt = huggingface_hub.hf_hub_download(spectral_shifts_ckpt, filename="spectral_shifts.safetensors", **hf_hub_kwargs)
        assert os.path.exists(spectral_shifts_ckpt)

        with safe_open(spectral_shifts_ckpt, framework="pt", device="cpu") as f:
            for key in f.keys():
                # spectral_shifts_weights[key] = f.get_tensor(key)
                accepts_dtype = "dtype" in set(inspect.signature(set_module_tensor_to_device).parameters.keys())
                if accepts_dtype:
                    set_module_tensor_to_device(model, key, param_device, value=f.get_tensor(key), dtype=torch_dtype)
                else:
                    set_module_tensor_to_device(model, key, param_device, value=f.get_tensor(key))
        print(f"Resumed from {spectral_shifts_ckpt}")
    if "torch_dtype"in kwargs:
        model = model.to(kwargs["torch_dtype"])
    model.register_to_config(_name_or_path=pretrained_model_name_or_path)
    # Set model in evaluation mode to deactivate DropOut modules by default
    model.eval()
    del original_model
    torch.cuda.empty_cache()
    return model



def load_text_encoder_for_svdiff(
        pretrained_model_name_or_path,
        spectral_shifts_ckpt=None,
        hf_hub_kwargs=None,
        **kwargs
    ):
    """
    https://github.com/huggingface/diffusers/blob/v0.14.0/src/diffusers/models/modeling_utils.py#L541
    """
    config = CLIPTextConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
    original_model = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
    state_dict = original_model.state_dict()
    with accelerate.init_empty_weights():
        model = CLIPTextModelForSVDiff(config)
    # load pre-trained weights
    param_device = "cpu"
    torch_dtype = kwargs["torch_dtype"] if "torch_dtype" in kwargs else None
    spectral_shifts_weights = {n: torch.zeros(p.shape) for n, p in model.named_parameters() if "delta" in n}
    state_dict.update(spectral_shifts_weights)
    # move the params from meta device to cpu
    missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
    if len(missing_keys) > 0:
        raise ValueError(
            f"Cannot load {model.__class__.__name__} from {pretrained_model_name_or_path} because the following keys are"
            f" missing: \n {', '.join(missing_keys)}. \n Please make sure to pass"
            " `low_cpu_mem_usage=False` and `device_map=None` if you want to randomely initialize"
            " those weights or else make sure your checkpoint file is correct."
        )

    for param_name, param in state_dict.items():
        accepts_dtype = "dtype" in set(inspect.signature(set_module_tensor_to_device).parameters.keys())
        if accepts_dtype:
            set_module_tensor_to_device(model, param_name, param_device, value=param, dtype=torch_dtype)
        else:
            set_module_tensor_to_device(model, param_name, param_device, value=param)
    
    if spectral_shifts_ckpt:
        if os.path.isdir(spectral_shifts_ckpt):
            spectral_shifts_ckpt = os.path.join(spectral_shifts_ckpt, "spectral_shifts_te.safetensors")
        elif not os.path.exists(spectral_shifts_ckpt):
            # download from hub
            hf_hub_kwargs = {} if hf_hub_kwargs is None else hf_hub_kwargs
            try:
                spectral_shifts_ckpt = huggingface_hub.hf_hub_download(spectral_shifts_ckpt, filename="spectral_shifts_te.safetensors", **hf_hub_kwargs)
            except huggingface_hub.utils.EntryNotFoundError:
                spectral_shifts_ckpt = None
        # load state dict only if `spectral_shifts_te.safetensors` exists
        if os.path.exists(spectral_shifts_ckpt):
            with safe_open(spectral_shifts_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    # spectral_shifts_weights[key] = f.get_tensor(key)
                    accepts_dtype = "dtype" in set(inspect.signature(set_module_tensor_to_device).parameters.keys())
                    if accepts_dtype:
                        set_module_tensor_to_device(model, key, param_device, value=f.get_tensor(key), dtype=torch_dtype)
                    else:
                        set_module_tensor_to_device(model, key, param_device, value=f.get_tensor(key))
            print(f"Resumed from {spectral_shifts_ckpt}")
        
    if "torch_dtype"in kwargs:
        model = model.to(kwargs["torch_dtype"])
    # model.register_to_config(_name_or_path=pretrained_model_name_or_path)
    # Set model in evaluation mode to deactivate DropOut modules by default
    model.eval()
    del original_model
    torch.cuda.empty_cache()
    return model



def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def slerp(val, low, high):
    """ taken from https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/4
    """
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res


def slerp_tensor(val, low, high):
    shape = low.shape
    res = slerp(val, low.flatten(1), high.flatten(1))
    return res.reshape(shape)


SCHEDULER_MAPPING = {
    "ddim": DDIMScheduler,
    "plms": PNDMScheduler,
    "lms": LMSDiscreteScheduler,
    "euler": EulerDiscreteScheduler,
    "euler_ancestral": EulerAncestralDiscreteScheduler,
    "dpm_solver++": DPMSolverMultistepScheduler,
}

