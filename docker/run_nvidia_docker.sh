#!/bin/bash

IMAGE_NAME=object_detection

xhost +

SCRIPT_DIR=$(cd $(dirname $0); pwd)

docker run -it --rm \
  --privileged \
  --runtime=nvidia \
  --env=QT_X11_NO_MITSHM=1 \
  --env="DISPLAY" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
  --net="host" \
  --volume="$SCRIPT_DIR/../:/root/$IMAGE_NAME/" \
  $IMAGE_NAME
