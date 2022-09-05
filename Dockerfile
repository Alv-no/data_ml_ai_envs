FROM nvidia/cuda:11.7.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Oslo
RUN apt-get update && apt upgrade -y

# install python3-pip
RUN apt-get install software-properties-common build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev -y
RUN wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz
RUN tar -xf Python-3.10.0.tgz
RUN cd Python-3.10.0/ && ./configure --enable-optimizations && make altinstall -j8
RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.10 1

RUN pip install --upgrade pip

RUN useradd -m -s /bin/bash docker_user
USER docker_user
ENV PATH "$PATH:/home/docker_user/.local/bin"
COPY ./requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt