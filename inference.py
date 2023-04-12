import argparse
import os
from tqdm import tqdm
import random
import torch
import huggingface_hub
from transformers import CLIPTextModel
from diffusers import StableDiffusionPipeline
from diffusers.utils import is_xformers_available
from svdiff_pytorch import load_unet_for_svdiff, load_text_encoder_for_svdiff, SCHEDULER_MAPPING, image_grid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, help="pretrained model name or path")
    parser.add_argument("--spectral_shifts_ckpt", type=str, help="path to spectral_shifts.safetensors")
    # diffusers config
    parser.add_argument("--prompt", type=str, nargs="?", default="a photo of *s", help="the prompt to render")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="number of sampling steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="unconditional guidance scale")
    parser.add_argument("--num_images_per_prompt", type=int, default=1, help="number of images per prompt")
    parser.add_argument("--height", type=int, default=512, help="image height, in pixel space",)
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space",)
    parser.add_argument("--seed", type=str, default="random_seed", help="the seed (for reproducible sampling)")
    parser.add_argument("--scheduler_type", type=str, choices=["ddim", "plms", "lms", "euler", "euler_ancestral", "dpm_solver++"], default="ddim", help="diffusion scheduler type")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--spectral_shifts_scale", type=float, default=1.0, help="scaling spectral shifts")
    parser.add_argument("--fp16", action="store_true", help="fp16 inference")
    args = parser.parse_args()
    return args


def load_text_encoder(pretrained_model_name_or_path, spectral_shifts_ckpt, device, fp16=False):
    if os.path.isdir(spectral_shifts_ckpt):
        spectral_shifts_ckpt = os.path.join(spectral_shifts_ckpt, "spectral_shifts_te.safetensors")
    elif not os.path.exists(spectral_shifts_ckpt):
        # download from hub
        hf_hub_kwargs = {} if hf_hub_kwargs is None else hf_hub_kwargs
        try:
            spectral_shifts_ckpt = huggingface_hub.hf_hub_download(spectral_shifts_ckpt, filename="spectral_shifts_te.safetensors", **hf_hub_kwargs)
        except huggingface_hub.utils.EntryNotFoundError:
            return CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=torch.float16 if fp16 else None).to(device)
    if not os.path.exists(spectral_shifts_ckpt):
            return CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=torch.float16 if fp16 else None).to(device)
    text_encoder = load_text_encoder_for_svdiff(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        spectral_shifts_ckpt=spectral_shifts_ckpt,
        subfolder="text_encoder", 
    )
    # first perform svd and cache
    for module in text_encoder.modules():
        if hasattr(module, "perform_svd"):
            module.perform_svd()
    if fp16:
        text_encoder = text_encoder.to(device, dtype=torch.float16)
    return text_encoder



def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    # load unet
    unet = load_unet_for_svdiff(args.pretrained_model_name_or_path, spectral_shifts_ckpt=args.spectral_shifts_ckpt, subfolder="unet")
    unet = unet.to(device)
    # first perform svd and cache
    for module in unet.modules():
        if hasattr(module, "perform_svd"):
            module.perform_svd()
    if args.fp16:
        unet = unet.to(device, dtype=torch.float16)
    text_encoder = load_text_encoder(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path, 
        spectral_shifts_ckpt=args.spectral_shifts_ckpt, 
        fp16=args.fp16,
        device=device
    )

    # load pipe
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet,
        text_encoder=text_encoder,
        requires_safety_checker=False,
        safety_checker=None,
        feature_extractor=None,
        scheduler=SCHEDULER_MAPPING[args.scheduler_type].from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler"),
        torch_dtype=torch.float16 if args.fp16 else None,
    )
    if args.enable_xformers_memory_efficient_attention:
        assert is_xformers_available()
        pipe.enable_xformers_memory_efficient_attention()
        print("Using xformers!")
    try:
        import tomesd
        tomesd.apply_patch(pipe, ratio=0.5)
        print("Using tomesd!")
    except:
        pass
    pipe = pipe.to(device)
    print("loaded pipeline")
    # run!
    if pipe.unet.conv_out.scale != args.spectral_shifts_scale:
        for module in pipe.unet.modules():
            if hasattr(module, "set_scale"):
                module.set_scale(scale=args.spectral_shifts_scale)
        if not isinstance(pipe.text_encoder, CLIPTextModel):
            for module in pipe.text_encoder.modules():
                if hasattr(module, "set_scale"):
                    module.set_scale(scale=args.spectral_shifts_scale)

        print(f"Set spectral_shifts_scale to {args.spectral_shifts_scale}!")
    
    if args.seed == "random_seed":
        random.seed()
        seed = random.randint(0, 2**32)
    else:
        seed = int(args.seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    print(f"seed: {seed}")
    prompts = args.prompt.split("::")
    all_images = []
    for prompt in tqdm(prompts):
        with torch.autocast(device), torch.inference_mode():
            images = pipe(
                prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                num_images_per_prompt=args.num_images_per_prompt,
                height=args.height,
                width=args.width,
            ).images
        all_images.extend(images)
    grid_image = image_grid(all_images, len(prompts), args.num_images_per_prompt)
    grid_image.save("grid.png")
    print("DONE! See `grid.png` for the results!")


if __name__ == '__main__':
    main()

