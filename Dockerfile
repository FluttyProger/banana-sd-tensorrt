FROM nvcr.io/nvidia/pytorch:23.02-py3

WORKDIR /

RUN apt-get update && apt-get install -y git wget

RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN python3 -m pip install --upgrade tensorrt

RUN git clone https://github.com/NVIDIA/TensorRT -b release/8.6

ADD app.py /TensorRT/demo/Diffusion

RUN ["ln", "-P", "/TensorRT/demo/Diffusion/app.py", "app.py"]

RUN mkdir -p /deliberate-model/engine

RUN wget -O /deliberate-model/engine/vae.plan https://huggingface.co/FluttyProger/Deliberate-onnx/resolve/main/engine/vae.plan

RUN wget -O /deliberate-model/engine/unet.plan https://huggingface.co/FluttyProger/Deliberate-onnx/resolve/main/engine/unet.plan

RUN wget -O /deliberate-model/engine/clip.plan https://huggingface.co/FluttyProger/Deliberate-onnx/resolve/main/engine/clip.plan

ADD txt2img_pipeline.py .

ADD stable_diffusion_pipeline.py .

ADD utilities.py .

ADD models.py .

RUN yes | /bin/mv txt2img_pipeline.py /TensorRT/demo/Diffusion

RUN yes | /bin/mv stable_diffusion_pipeline.py /TensorRT/demo/Diffusion

RUN yes | /bin/mv utilities.py /TensorRT/demo/Diffusion

RUN yes | /bin/mv models.py /TensorRT/demo/Diffusion

EXPOSE 8000

ENV CUDA_MODULE_LOADING=LAZY

CMD python3 -u app.py
