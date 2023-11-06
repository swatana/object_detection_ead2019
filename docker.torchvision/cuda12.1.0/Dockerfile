FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                       apt-utils \
                       git \
                       curl \
                       python3 \
                       python3-pip \
                       python3-setuptools \
                       wget \
                       libopencv-dev \
                       build-essential

RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install numpy\
                 matplotlib\
                 pandas\
                 Pillow\
                 torch==2.1.0\
                 torchvision==0.16.0\
                 opencv-python

# nvidia environment
# for 2
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

RUN apt-get update && \
    apt-get install -y --no-install-recommends mesa-utils x11-apps && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE 1

WORKDIR /root
