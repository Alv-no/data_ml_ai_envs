FROM nvidia/cuda:11.7.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Oslo
RUN apt-get update && apt upgrade -y
ARG DEBIAN_FRONTEND=noninteractive
ARG OPENCV_VERSION=4.6.0
ARG PYTHON_VERSION=3.10.4

RUN apt-get update && apt-get upgrade -y &&\
    apt-get install -y \
    wget software-properties-common build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev libbz2-dev git -y

RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz
RUN tar -xf Python-${PYTHON_VERSION}.tgz
RUN cd Python-${PYTHON_VERSION}/ && ./configure --enable-optimizations && make altinstall -j"$(nproc)"
RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.10 1

RUN pip install --upgrade pip
RUN pip install numpy

RUN apt-get update && apt-get upgrade -y &&\
    # Install build tools, build dependencies and python
    apt-get install -y \
	# python3-pip \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
        libxine2-dev \
        libglew-dev \
        libtiff5-dev \
        zlib1g-dev \
        libjpeg-dev \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libpostproc-dev \
        libswscale-dev \
        libeigen3-dev \
        libtbb-dev \
        libgtk2.0-dev \
        pkg-config \
        ## Python
        # python3-dev \
        # python3-numpy \
    && rm -rf /var/lib/apt/lists/*

RUN python --version

RUN cd /opt/ &&\
    # Download and unzip OpenCV and opencv_contrib and delte zip files
    wget https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip &&\
    unzip $OPENCV_VERSION.zip &&\
    rm $OPENCV_VERSION.zip &&\
    wget https://github.com/opencv/opencv_contrib/archive/$OPENCV_VERSION.zip &&\
    unzip ${OPENCV_VERSION}.zip &&\
    rm ${OPENCV_VERSION}.zip &&\
    # Create build folder and switch to it
    mkdir /opt/opencv-${OPENCV_VERSION}/build && cd /opt/opencv-${OPENCV_VERSION}/build &&\
    # Cmake configure
    cmake \
        -DOPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib-${OPENCV_VERSION}/modules \
        # Additional
        -DPYTHON_DEFAULT_EXECUTABLE=$(which python) \
        -DBUILD_NEW_PYTHON_SUPPORT=ON \
        -DBUILD_opencv_python3=ON \
        -DOPENCV_DNN_CUDA=ON \
        -DHAVE_opencv_python3=ON \
        -D WITH_TBB=ON \
        -DWITH_OPENMP=ON \
        -DWITH_IPP=ON \
        -DWITH_CSTRIPES=ON \
        -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")  \
        -DPYTHON3_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
        -DPYTHON3_LIBRARY=$(python -c "from distutils.sysconfig import get_config_var;from os.path import dirname,join ; print(join(dirname(get_config_var('LIBPC')),get_config_var('LDLIBRARY')))")  \
        -DPYTHON3_NUMPY_INCLUDE_DIRS=$(python -c "import numpy; print(numpy.get_include())")  \
        -DPYTHON3_PACKAGES_PATH=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")  \
        -DOPENCV_GENERATE_PKGCONFIG=ON \
        -Dopencv_dnn_superres=ON \
        # Min functional
        -DWITH_CUDA=ON \
        -DWITH_CUDNN=ON \
        -DCUDA_ARCH_BIN=7.5,8.0 \
        -DCMAKE_BUILD_TYPE=RELEASE \
        # Install path will be /usr/local/lib (lib is implicit)
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        .. &&\
    # Make
    make -j"$(nproc)" && \
    # Install to /usr/local/lib
    make install && \
    ldconfig &&\
    # Remove OpenCV sources and build folder
    rm -rf /opt/opencv-${OPENCV_VERSION} && rm -rf /opt/opencv_contrib-${OPENCV_VERSION}



RUN useradd -m -s /bin/bash docker_user

USER docker_user
ENV PATH "$PATH:/home/docker_user/.local/bin"
COPY ./requirements.txt ./requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt


ENV PYTHONPATH "${PYTHONPATH}:/home/docker_user/tensorflow_environment/tests:/home/docker_user/tensorflow_environment:"
WORKDIR /home/docker_user/tensorflow_environment
USER root
RUN usermod -a -G video docker_user
RUN apt-get update && apt-get upgrade -y && apt-get install libcanberra-gtk-module libcanberra-gtk3-module -y
USER docker_user
ENTRYPOINT ["bash"]