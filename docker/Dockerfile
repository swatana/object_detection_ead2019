FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

RUN DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                       apt-utils \
                       git \
                       python3-dev \
                       python3-pip \
                       python3-setuptools \
                       curl \
                       wget \
                       libopencv-dev

RUN pip3 install --upgrade pip

RUN pip3 install numpy \
                 matplotlib \
                 tqdm \
                 tensorflow-gpu==1.9.0 \
                 tensorboardX \
                 keras \
                 opencv-python \
                 pillow \
                 scikit-learn \
                 scikit-image

# nvidia environment
# for 2
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

RUN apt-get update && \
    apt-get install -y --no-install-recommends mesa-utils x11-apps && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE 1

WORKDIR /root
