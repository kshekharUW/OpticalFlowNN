# Start FROM PyTorch image https://hub.docker.com/r/pytorch/pytorch or nvcr.io/nvidia/pytorch:23.03-py3
# cd /mnt/c/Users/Krishnendu/Documents/GIT/Raft/Docker/
# SHELL ["/bin/bash", "-c"]	

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime


# ARG PYTORCH="1.6.0"
# ARG CUDA="10.1"
# ARG CUDNN="7"
# FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
# # https://github.com/NVIDIA/nvidia-container-toolkit/issues/258#issuecomment-1903944103
# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list
# RUN apt-key del 7fa2af80
# RUN apt-get update && apt-get install -y --no-install-recommends wget
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
# RUN dpkg -i cuda-keyring_1.0-1_all.deb


RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

# Install pip packages
# RUN python3 -m pip install --upgrade pip wheel
RUN pip install --no-cache matplotlib 
RUN pip install --no-cache tensorboard 
RUN pip install --no-cache scipy 
RUN pip install --no-cache opencv-python


# RUN conda init bash
# RUN conda create --name raftEnv -y
# SHELL ["conda", "run", "-n", "raftEnv", "/bin/bash", "-c"]
# RUN conda activate raft
# RUN conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 matplotlib tensorboard scipy opencv -c pytorch

# Set environment variables
ENV OMP_NUM_THREADS=1
# Avoid DDP error "MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library" https://github.com/pytorch/pytorch/issues/37377
ENV MKL_THREADING_LAYER=GNU

# As the user, make the directory we will map the volume to, as the user, so it has proper permissions
RUN mkdir -p /home/volume
WORKDIR /home/volume/RAFT
ENV FORCE_CUDA="1"