FROM nvcr.io/nvidia/pytorch:23.02-py3

WORKDIR /

RUN apt-get update && apt-get install -y git wget

RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN git clone https://github.com/NVIDIA/TensorRT -b release/8.6

ADD app.py /TensorRT/demo/Diffusion

RUN mkdir -p /deliberate-model/engine

RUN wget -O /deliberate-model/engine/vae.plan https://huggingface.co/FluttyProger/Deliberate-onnx/resolve/main/engine/vae.plan

RUN wget -O /deliberate-model/engine/unet.plan https://huggingface.co/FluttyProger/Deliberate-onnx/resolve/main/engine/unet.plan

RUN wget -O /deliberate-model/engine/clip.plan https://huggingface.co/FluttyProger/Deliberate-onnx/resolve/main/engine/clip.plan

EXPOSE 8000

CMD python3 -u /TensorRT/demo/Diffusion/app.py
