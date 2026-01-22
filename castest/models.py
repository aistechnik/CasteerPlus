import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel


def get_model(model):
    if model == 'sdxl':
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            cache_dir='./cache'
            )
    elif model == 'sdxl-turbo':
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir='./cache'
            )
    elif model == 'sdxl-tuned':
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            cache_dir='./cache'
            )

        # load finetuned model
        unet = UNet2DConditionModel.from_pretrained(
            "mhdang/dpo-sdxl-text2image-v1",
            subfolder="unet",
            torch_dtype=torch.float16)

        pipe.unet = unet
   
    elif model == 'sdxl-turbo-tuned':
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            cache_dir='./cache'
            )

        # load finetuned model
        unet = UNet2DConditionModel.from_pretrained(
            "mhdang/dpo-sdxl-text2image-v1",
            subfolder="unet",
            torch_dtype=torch.float16)

        pipe.unet = unet

    return pipe
