#!/bin/bash

nvidia-docker build -t andq-training .

nvidia-docker run --name hand-pose-training \
		-it --ipc=host --rm \
		-p 8888:8888 \
		-v /home/andq/yolov5/:/usr/src/yolo5 \
		-v /media/local-data/andq/dex-vcb/:/usr/src/dex-vcb \
		andq-training bash # image name
