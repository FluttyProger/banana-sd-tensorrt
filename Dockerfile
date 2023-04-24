FROM runpod/pytorch:3.10-2.0.0-117

WORKDIR /

RUN apt-get update && apt-get install -y git wget

RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN python3 -m pip install --upgrade tensorrt
RUN python3 -m pip install --upgrade polygraphy onnx-graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com
RUN python3 -m pip install onnxruntime

RUN git clone https://github.com/FluttyProger/diffusers

WORKDIR /diffusers

RUN pip install .

WORKDIR /

ARG MODEL_NAME
ENV MODEL_NAME=stabilityai/stable-diffusion-2-1

ARG MODEL_REV
ENV MODEL_REV=fp16

ADD app.py .

RUN wget -O /usr/local/lib/python3.10/dist-packages/torch/onnx/_constants.py https://raw.githubusercontent.com/pytorch/pytorch/d06d195bcd960f530f8f0d5a1992ed68d2823d4e/torch/onnx/_constants.py

RUN wget -O /usr/local/lib/python3.10/dist-packages/torch/onnx/symbolic_opset14.py https://raw.githubusercontent.com/pytorch/pytorch/d06d195bcd960f530f8f0d5a1992ed68d2823d4e/torch/onnx/symbolic_opset14.py

ADD download.py .
RUN python3 download.py

RUN wget /root/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1/snapshots/f7f33030acc57428be85fbec092c37a78231d75a/engine/vae.plan https://huggingface.co/FluttyProger/stable-diffusion-2-1-tensorrt/resolve/main/vae.plan

RUN wget /root/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1/snapshots/f7f33030acc57428be85fbec092c37a78231d75a/engine/unet.plan https://huggingface.co/FluttyProger/stable-diffusion-2-1-tensorrt/resolve/main/unet.plan

RUN wget /root/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1/snapshots/f7f33030acc57428be85fbec092c37a78231d75a/engine/clip.plan https://huggingface.co/FluttyProger/stable-diffusion-2-1-tensorrt/resolve/main/clip.plan


EXPOSE 8000

CMD python3 -u app.py
