# Base image - update if need a different base
FROM nvidia/cudagl:10.1-devel-ubuntu16.04

# Setup basic packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    vim \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libglfw3-dev \
    libglm-dev \
    libx11-dev \
    libomp-dev \
    libegl1-mesa-dev \
    pkg-config \
    wget \
    zip \
    unzip &&\
    rm -rf /var/lib/apt/lists/*

# Install conda
RUN curl -L -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN chmod +x ~/miniconda.sh
RUN ~/miniconda.sh -b -p /opt/conda
RUN rm ~/miniconda.sh
RUN /opt/conda/bin/conda install numpy pyyaml scipy ipython mkl mkl-include
RUN /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

# Install cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.14.0/cmake-3.14.0-Linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh /cmake-3.14.0-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version

# Conda environment
RUN conda create -n habitat python=3.6 cmake=3.14.0

# Setup habitat-sim
RUN git clone --branch v0.1.7 https://github.com/facebookresearch/habitat-sim.git
RUN /bin/bash -c ". activate habitat; cd habitat-sim; pip install -r requirements.txt; python setup.py build_ext --parallel 2 install --headless --with-cuda"

# Install habitat-lab
RUN git clone --branch v0.1.7 https://github.com/facebookresearch/habitat-lab.git
RUN /bin/bash -c ". activate habitat; cd habitat-lab; pip install --no-cache-dir h5py; pip install -r requirements.txt; python setup.py develop --all"

# Silence habitat-sim logs
ENV GLOG_minloglevel=2
ENV MAGNUM_LOG="quiet"

# add our code requirements
ADD requirements.txt requirements.txt
RUN /bin/bash -c ". activate habitat; pip install -r requirements.txt"

# Add code
ADD configs configs
ADD datasets datasets
ADD models models
ADD planning planning
ADD pytorch_utils pytorch_utils
ADD main.py main.py
ADD metrics.py metrics.py
ADD store_episodes_parallel.py store_episodes_parallel.py
ADD store_img_segm_ep.py store_img_segm_ep.py
ADD test_utils.py test_utils.py
ADD tester.py tester.py
ADD train_options.py train_options.py
ADD trainer_active.py trainer_active.py
ADD trainer_finetune.py trainer_finetune.py
ADD trainer_segm.py trainer_segm.py
ADD trainer.py trainer.py
