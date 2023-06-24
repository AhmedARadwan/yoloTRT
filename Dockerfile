ARG TENSORRT_TAR=TensorRT-7.2.2.3.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.0.tar.gz

FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04
RUN apt update

ENV DEBIAN_FRONTEND noninteractive
ENV TENSORRT_INSTALL=/app/TensorRT-7.2.2.3

# Install common build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    autoconf \
    automake \
    build-essential \
    curl \
    dialog apt-utils \
    git \
    g++ \
    libboost-all-dev \
    libssl-dev \
    libtool \
    libyaml-cpp-dev \
    lsb-release \
    make \
    pkg-config \
    python3-pip \
    python3-setuptools \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install python packages
RUN pip3 install cmake==3.18.0

# Copy Nvidia sources
COPY TensorRT /app/TensorRT/

# tensorrt installation
COPY TensorRT-7.2.2.3.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.0.tar.gz /app/
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/app/TensorRT-7.2.2.3/lib
RUN tar -xvzf /app/TensorRT-7.2.2.3.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.0.tar.gz -C /app/ \
    && rm /app/TensorRT-7.2.2.3.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.0.tar.gz \
    && export TRT_LIBPATH=/app/TensorRT-7.2.2.3 \
    # Next build steps are done in TensorRT repo
    && cd /app/TensorRT/ \
    && export TRT_SOURCE=`pwd` \
    && mkdir -p build && cd build \
    && cmake .. \
        -DTRT_LIB_DIR=$TRT_LIBPATH/lib \
        -DTRT_OUT_DIR=`pwd`/out \
        -DCUDA_VERISON=11.3.0 \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SAMPLES=OFF \
    && make -j$(nproc)

# Install ROS melodic on Ubuntu (http://wiki.ros.org/melodic/Installation/Ubuntu)
ENV ROS_DISTRO=melodic
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN apt-get update && apt-get install -y ros-${ROS_DISTRO}-ros-base
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc
RUN apt update && apt install -y python3-catkin-tools \
                                 python-rosdep \
                                 python-rosinstall \
                                 python-rosinstall-generator \
                                 python-wstool \
                                 build-essential \
                                 libopencv-dev \
                                 ros-${ROS_DISTRO}-cv-bridge \
                                 ros-${ROS_DISTRO}-video-stream-opencv \
                                 ros-${ROS_DISTRO}-rviz


# copy workspace inside docker
COPY ros /home/ros

ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

WORKDIR /home/ros/

     

