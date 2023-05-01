import os
import base64
from io import BytesIO
import tensorrt as trt
from sanic import Sanic, response
from utilities import TRT_LOGGER, add_arguments
from txt2img_pipeline import Txt2ImgPipeline

model = None

server = Sanic("my_app")

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

trt.init_libnvinfer_plugins(TRT_LOGGER, '')
model = Txt2ImgPipeline(
    output_dir="",
    version="1.5",
    hf_token="",
    max_batch_size=4)
model.loadEngines("/deliberate-model/engine")

@server.route('/healthcheck', methods=["GET"])
def healthcheck(request):
    # dependency free way to check if GPU is visible
    gpu = False
    out = subprocess.run("nvidia-smi", shell=True)
    if out.returncode == 0: # success state on shell command
        gpu = True

    return response.json({"state": "healthy", "gpu": gpu})

@server.route('/', methods=["POST"]) 
def handler(request):
    global model
    try:
        model_inputs = response.json.loads(request.json)
    except:
        model_inputs = request.json
    prompt = model_inputs.get('prompt', None)
    height = model_inputs.get('height', 768)
    negative = model_inputs.get('negative_prompt', None)
    width = model_inputs.get('width', 768)
    steps = model_inputs.get('steps', 36)
    guidance_scale = model_inputs.get('guidance_scale', 7)
    seed = model_inputs.get('seed', -1)
    sampler = model_inputs.get('sampler', "DDIM")

    if not prompt: return response.json({'message': 'No prompt was provided'})


    # Load TensorRT engines and pytorch modules
    model.loadResources(height, width, 1, seed, steps, guidance_scale, sampler)


    images = model.infer([prompt], negative_prompt=[negative], image_height=height, image_width=width, seed=seed)
    
    buffered = BytesIO()
    images[0].save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return response.json({"outputs": outputs})


if __name__ == "__main__":
    server.run(host='0.0.0.0', port=8000, workers=1)
