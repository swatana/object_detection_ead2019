#!/bin/bash

IMAGE_NAME=torchvision016

xhost +

SCRIPT_DIR=$(cd $(dirname $0); pwd)

docker run -it --rm \
  --privileged \
  --gpus all \
  --env=QT_X11_NO_MITSHM=1 \
  --env="DISPLAY" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
  --net="host" \
  --volume="$SCRIPT_DIR/../:/root/object_detection_ead2019/" \
  $IMAGE_NAME
