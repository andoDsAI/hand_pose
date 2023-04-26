#!/bin/bash
set -exu
nvidia-docker build --tag local/oru-dgx -f Dockerfile ./
