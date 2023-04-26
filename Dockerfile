FROM nvcr.io/nvidia/pytorch:20.12-py3

# Install linux packages
RUN apt-get update \
    && \
    echo "------------------------------------------------------ essentials" \
    && \
    apt-get install -y --no-install-recommends -y \
    build-essential \
    apt-utils \
    && \
    echo "------------------------------------------------------ editors" \
    && \
    apt-get install -y --no-install-recommends -y \
    emacs \
    vim \
    nano \
    && \
    echo "------------------------------------------------------ software" \
    && \
    apt-get install -y --no-install-recommends -y \
    python3-pip \
    tmux \ 
    && \
    echo "------------------------------------------------------ cleanup" \
    apt-get clean && rm -rf /var/lib/apt/lists/* \
	&& apt-get install -y --no-install-recommends -y \
	screen \
	libgl1-mesa-glx \
	libglfw3-dev \
	libgles2-mesa-dev

# Install python dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip3 install -r requirements.txt

# Create working directory in image container
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy all contents to image container
COPY . /usr/src/app
