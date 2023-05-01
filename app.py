import os
import base64
from io import BytesIO
from potassium import Potassium, Request, Response
import tensorrt as trt
from utilities import TRT_LOGGER, add_arguments
from txt2img_pipeline import Txt2ImgPipeline

app = Potassium("my_app")

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

@app.init
def init():
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    demo = Txt2ImgPipeline(
        output_dir="",
        version="1.5",
        hf_token="",
        max_batch_size=4)
    demo.loadEngines("/deliberate-model/engine")
    context = {
        "model": demo
    }
    return context


@app.handler()
def handler(context, request: Request) -> Response:
    model = context.get("model")
    model_inputs = request.json
    outputs = inference(model, model_inputs)
    
    return Response(json={"outputs": outputs}, status=200)


# Inference is ran for every server call
# Reference your preloaded global model variable here.


def inference(model, model_inputs: dict) -> dict:
    prompt = model_inputs.get('prompt', None)
    height = model_inputs.get('height', 768)
    negative = model_inputs.get('negative_prompt', None)
    width = model_inputs.get('width', 768)
    steps = model_inputs.get('steps', 36)
    guidance_scale = model_inputs.get('guidance_scale', 7)
    seed = model_inputs.get('seed', -1)

    if not prompt: return {'message': 'No prompt was provided'}


    # Load TensorRT engines and pytorch modules
    model.loadResources(height, width, 1, seed, steps, guidance_scale, "DDIM")


    images = model.infer([prompt], negative_prompt=[negative], image_height=height, image_width=width, seed=seed)
    
    buffered = BytesIO()
    images[0].save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return {'image_base64': image_base64}


if __name__ == "__main__":
    app.serve(host="127.0.0.1")
