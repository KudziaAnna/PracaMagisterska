FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
MAINTAINER Sano - Centre for Computational Medicine
WORKDIR /feta

ADD requirements.txt .
RUN pip install -r requirements.txt

ADD models/model-ModifiedSwinUNet-fold-0.pt .
ADD models/model-ModifiedSwinUNet-fold-1.pt .
ADD models/model-ModifiedSwinUNet-fold-2.pt .
ADD models/model-ModifiedSwinUNet-fold-3.pt .
ADD models/model-ModifiedSwinUNet-fold-4.pt .
ADD src/models ./models
ADD src/inference.py .

