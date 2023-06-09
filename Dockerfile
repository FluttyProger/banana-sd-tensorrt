FROM nvcr.io/nvidia/pytorch:23.02-py3

WORKDIR /

RUN apt-get update && apt-get install -y git wget

RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN python3 -m pip install --upgrade tensorrt

ADD app.py .

RUN mkdir -p /deliberate-model/engine

RUN wget -O /deliberate-model/engine/vae.plan https://huggingface.co/FluttyProger/Deliberate-onnx/resolve/main/engine/vae.plan

RUN wget -O /deliberate-model/engine/unet.plan https://huggingface.co/FluttyProger/Deliberate-onnx/resolve/main/engine/unet.plan

RUN wget -O /deliberate-model/engine/clip.plan https://huggingface.co/FluttyProger/Deliberate-onnx/resolve/main/engine/clip.plan

ADD txt2img_pipeline.py .

ADD stable_diffusion_pipeline.py .

ADD utilities.py .

ADD models.py .

ADD text_encoder.py .

ADD lpw.py .

ADD special_tokens_map.json /files/tokenizer/

ADD vocab.json /files/tokenizer/

ADD tokenizer_config.json /files/tokenizer/

ADD merges.txt /files/tokenizer/

ADD scheduler_config.json /files/scheduler/

EXPOSE 8000

ENV CUDA_MODULE_LOADING=LAZY

CMD python3 -u app.py
