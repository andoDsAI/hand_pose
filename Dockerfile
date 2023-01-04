FROM nvcr.io/nvidia/pytorch:20.12-py3

# Install linux packages
RUN apt-get update && apt-get install -y --no-install-recommends screen libgl1-mesa-glx

# Install python dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip install -r requirements.txt gsutil

# Create working directory in image container
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy all contents to image container
COPY . /usr/src/app
