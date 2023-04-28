import os
import torch
from torch import autocast
import base64
from io import BytesIO
from transformers import pipeline
from diffusers import DDIMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from potassium import Potassium, Request, Response


app = Potassium("my_app")


@app.init
def init():
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
    
    model = model.to("cuda")
    model.enable_xformers_memory_efficient_attention()
    context = {
        "model": model
    }
    
    return context


@app.handler()
def handler(context: dict, request: Request) -> Response:
    model_inputs = request.json
    model = context.get("model")
    outputs = inference(model, model_inputs)
    
    return Response(json={"outputs": outputs}, status=200)


# Inference is ran for every server call
# Reference your preloaded global model variable here.


def inference(model, model_inputs: dict) -> dict:
    prompt = model_inputs.get('prompt', None)
    height = model_inputs.get('height', 768)
    negative = model_inputs.get('negative_prompt', None)
    width = model_inputs.get('width', 768)
    steps = model_inputs.get('steps', 20)
    guidance_scale = model_inputs.get('guidance_scale', 9)
    seed = model_inputs.get('seed', None)

    if not prompt: return {'message': 'No prompt was provided'}
    
    generator = None
    if seed: generator = torch.Generator("cuda").manual_seed(seed)
    
    with autocast("cuda"):
        image = model(prompt, negative_prompt=negative, guidance_scale=guidance_scale, height=height, width=width, num_inference_steps=steps, generator=generator)
    
    buffered = BytesIO()
    image.images[0].save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return {'image_base64': image_base64}


if __name__ == "__main__":
    app.serve()
