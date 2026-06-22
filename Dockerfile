# Use an NVIDIA CUDA base image that includes development libraries
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# path to the local OptiX 7.6 from https://developer.nvidia.com/designworks/optix/downloads/legacy
# put your files there or set the arg during docker build
ARG LOCAL_OPTIX_DIRECTORY=optix/

# Non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# ------------------------------------------------------
# 1) Install System Dependencies
# ------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    cmake \
    unzip \
    build-essential \
    libglew-dev \
    libassimp-dev \
    libboost-all-dev \
    libgtk-3-dev \
    libopencv-dev \
    libglfw3-dev \
    libavdevice-dev \
    libavcodec-dev \
    libeigen3-dev \
    libxxf86vm-dev \
    libembree-dev \
    # libabsl-dev \
    libcgal-dev \
    libglm-dev \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------
# 2) Install a Miniconda environment
# ------------------------------------------------------
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Make conda available and create environment
ENV PATH="/opt/conda/bin:${PATH}"
ENV CONDA_PLUGINS_AUTO_ACCEPT_TOS=true
RUN conda update -n base -c defaults conda && \
    conda create -n ever python=3.10 -y && \
    conda clean -ya

SHELL ["/bin/bash", "-c"]
WORKDIR /ever_training

# ------------------------------------------------------
# 3) Install Python packages (within the 'ever' env)
# ------------------------------------------------------
RUN source activate ever && \
    # Adjust the PyTorch install line for your specific CUDA version if needed
    pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

COPY ./requirements.txt .
RUN source activate ever && \
    # this pip version seems to install pytorch3d successfully, a newer one (25.3) fails
    pip install pip==23.3 && \
    pip install --no-cache-dir tensorboard && \
    pip install --no-cache-dir setuptools==69.5.1 && \
    pip install --no-cache --no-cache-dir -r requirements.txt
RUN rm ./requirements.txt

ENV TORCH_CUDA_ARCH_LIST="5.0;6.0;6.1;7.0;7.5;8.0;8.6"

COPY submodules/simple-knn submodules/simple-knn
RUN source activate ever && \
    pip install --no-build-isolation submodules/simple-knn/

# ------------------------------------------------------
# 4) Install Slang
# ------------------------------------------------------
RUN mkdir /slang_install && \
    cd /slang_install && \
    wget https://github.com/shader-slang/slang/releases/download/v2025.6.1/slang-2025.6.1-linux-x86_64.zip && \
    unzip slang-2025.6.1-linux-x86_64.zip && \
    cp bin/* /usr/bin/
ENV LD_LIBRARY_PATH="/slang_install/lib/"

# Clone, build, and install abseil-cpp.
RUN git clone https://github.com/abseil/abseil-cpp.git /tmp/abseil-cpp && \
    cd /tmp/abseil-cpp && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    rm -rf /tmp/abseil-cpp

# ------------------------------------------------------
# 5) Copy OptiX
# ------------------------------------------------------
    
# Set an environment variable for OptiX installation.
ENV OptiX_INSTALL_DIR=/opt/OptiX
COPY $LOCAL_OPTIX_DIRECTORY /opt/OptiX
RUN ls /opt/OptiX

# ------------------------------------------------------
# 6) Final Container Setup
# ------------------------------------------------------

ENV CUDAARCHS="50 60 61 70 75 80 86"

# Expose any ports needed for training or viewer
EXPOSE 6009

# By default, just start a shell in the 'ever' environment
CMD ["/bin/bash", "-c", "source activate ever && exec bash"]

