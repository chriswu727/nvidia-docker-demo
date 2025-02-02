#use base images of pytorch
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

#install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install matplotlib numpy

COPY train.py /app/train.py
WORKDIR /app

CMD ["python", "train.py"]