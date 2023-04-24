import cv2
import torch
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from diffusers.utils import load_image
from diffusers import UniPCMultistepScheduler, StableDiffusionControlNetPipeline, ControlNetModel
from potassium import Potassium, Request, Response
from torchvision import transforms


app = Potassium("my_app")


@app.init
def init():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"using device {dev}")
    device = torch.device(dev)

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    model = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float16).to(device)
    model.scheduler = UniPCMultistepScheduler.from_config(
        model.scheduler.config)
    # https://github.com/huggingface/diffusers/issues/2907
    model.enable_model_cpu_offload()
    model.enable_xformers_memory_efficient_attention()
    context = {
        "model": model
    }

    return context


@app.handler()
def handler(context: dict, request: Request) -> Response:
    model_inputs = request.json
    model = context.get("model")
    controlnet = context.get("controlnet")
    outputs = inference(model, model_inputs)
    
    return Response(json={"outputs": outputs}, status=200)


# Inference is ran for every server call
# Reference your preloaded global model variable here.


def inference(model, model_inputs: dict) -> dict:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    negative_prompt = model_inputs.get('negative_prompt', None)
    num_inference_steps = model_inputs.get('num_inference_steps', 20)
    image_data = model_inputs.get('image_data', None)
    if prompt == None:
        return {'message': "No prompt provided"}

    # Run the model
    image = Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB")
    image = np.array(image)
    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)

    canny_image = Image.fromarray(image)
    buffered = BytesIO()
    canny_image.save(buffered, format="JPEG")
    canny_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

   
    output = model(prompt,
                   canny_image,
                   negative_prompt=negative_prompt,
                   num_inference_steps=5,)

    image = output.images[0]
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Return the results as a dictionary
    return {'canny_base64': canny_base64, 'image_base64': image_base64}

if __name__ == "__main__":
    app.serve()