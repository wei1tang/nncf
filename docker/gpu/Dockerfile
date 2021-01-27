FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

RUN echo "deb https://mirrors.aliyun.com/nvidia-cuda/ubuntu1804/x86_64/ ./" > /etc/apt/sources.list.d/cuda.list

#RUN curl -fsSL https://mirrors.aliyun.com/nvidia-cuda/ubuntu1804/x86_64/7fa2af80.pub | apt-key add -
# Download https://mirrors.aliyun.com/nvidia-cuda/ubuntu1804/x86_64/7fa2af80.pub and save it to nncf/docker/gpu
COPY ./7fa2af80.pub /tmp/7fa2af80.pub
RUN apt-key add /tmp/7fa2af80.pub

RUN echo "" > /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse" >> /etc/apt/sources.list&& \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-security main restricted universe multiverse" >> /etc/apt/sources.list

RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-transport-https \
        git && \
rm -rf /var/lib/apt/lists/*

# Required for nvidia-docker v1
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf


RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
        wget \
        curl \
        zip \
        unzip \
        nano \
        openssh-server \
        openssh-client \
        sudo \
		python3 \
		python3-dev \
		python3-pip &&\
	cd /usr/bin && \
	ln -s python3.8 python && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install tensorflow-gpu && \
    pip3 uninstall -y protobuf && \
    pip3 install google \
        protobuf \
        tensorflow_datasets \
        tensorflow-addons \
        pycocotools \
        opencv-python

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

ARG https_proxy
ARG no_proxy
ENV https_proxy=$https_proxy
ENV no_proxy=$no_proxy

ENTRYPOINT cd /home/nncf/ && python setup.py install && bash
