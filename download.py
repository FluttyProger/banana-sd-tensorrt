# In this file, we define download_model
# It runs during container build time to get model weights built into the container
import os
import torch
from torch import autocast
import base64
from io import BytesIO
from transformers import pipeline
from diffusers import DDIMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline

def download_model():
    model_name = os.getenv("MODEL_NAME")
    model_rev = os.getenv("MODEL_REV")
    scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
    
    model = StableDiffusionPipeline.from_pretrained(model_name,
                                                    custom_pipeline="stable_diffusion_tensorrt_txt2img_nobuild",
                                                    revision=model_rev,
                                                    torch_dtype=torch.float16,
                                                    scheduler=scheduler)
    
    # re-use cached folder to save ONNX models and TensorRT Engines
    model.set_cached_folder(model_name, revision=model_rev)


if __name__ == "__main__":
    download_model()
