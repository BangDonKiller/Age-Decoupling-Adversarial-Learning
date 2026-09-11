FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y git libsndfile1

RUN pip install --no-cache-dir -r requirements.txt

RUN pip uninstall -y pysoundfile soundfile \
 && pip install soundfile>=0.11

RUN apt-get update && apt-get install -y ffmpeg