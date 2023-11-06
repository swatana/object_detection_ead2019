FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                       apt-utils \
                       git \
                       python3-dev \
                       python3-pip \
                       python3-setuptools \
                       curl \
                       wget \
                       libopencv-dev \
                       build-essential

RUN pip3 install --upgrade pip

RUN pip3 install -vvv \
                 numpy \
                 matplotlib \
                 tqdm \
                 tensorflow-gpu==1.15.5 \
                 tensorboardX \
                 keras==2.1.6 \
                 opencv-python \
                 pillow \
                 scikit-learn \
                 scikit-image \
                 imgaug

# nvidia environment
# for 2
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

RUN apt-get update && \
    apt-get install -y --no-install-recommends mesa-utils x11-apps && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE 1

WORKDIR /root
