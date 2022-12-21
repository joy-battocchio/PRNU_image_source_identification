FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

# Install base utilities
RUN apt-get update && \
    apt install -y software-properties-common ca-certificates wget curl ssh && \
    apt-get install -y build-essential  && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update
RUN apt-get install git -y
RUN apt-get install nano -y
RUN apt-get install bash-completion -y
RUN apt-get install unzip -y

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda update -n base -c defaults conda
RUN conda init bash

# setup cnn fast sdi repo
WORKDIR /~
RUN git clone https://github.com/polimi-ispl/cnn-fast-sdi.git
WORKDIR /~/cnn-fast-sdi
RUN conda env create -f environment.yml
RUN echo "conda activate cnn_fast_sdi" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# if you want to import a model downloaded from here (https://github.com/polimi-ispl/cnn-fast-sdi) uncomment the following lines
# RUN mkdir downloaded_pretrained_models
# ADD cropr224_Pcn /~/cnn-fast-sdi/downloaded_pretrained_models/cropr224_Pcn/

RUN /opt/conda/envs/cnn_fast_sdi/bin/pip uninstall h5py -y
RUN /opt/conda/envs/cnn_fast_sdi/bin/pip install --upgrade h5py==2.10.0

WORKDIR /~/
ADD Demo.ipynb /~/
# setup prnu repo - Not used for the cnn-tests
# WORKDIR /~
# RUN git clone https://github.com/polimi-ispl/prnu-python.git
# WORKDIR /~/prnu-python
# RUN /opt/conda/envs/cnn_fast_sdi/bin/pip install -r requirements.txt

# Uncomment if you downloaded the data folder on the local directory
# Download data from https://drive.google.com/drive/folders/1TPLPCwCuQgmRslpjd-e5syqq7JK3DfVw?usp=sharing before doing this
# WORKDIR /~/
# ADD data/Vision_Frontal_Separated/Vision /~/cnn-fast-sdi/Vision/
# ADD data/Vision_Frontal_Separated/Noises_lists /~/cnn-fast-sdi/Noises_lists/

# If you want to download VISION uncomment the following 2 lines
# WORKDIR /~/
# RUN wget https://lesc.dinfo.unifi.it/VISION/VISION_base_files.txt && wget -x -nv -i VISION_base_files.txt

# sudo docker build -t cnn_fast_sdi_runtime .
# sudo docker run -it --gpus all cnn_fast_sdi_runtime /bin/bash
# sudo docker run -it --gpus all -v /path/in/host/machine:/~/shared cnn_fast_sdi_runtime /bin/bash
# python3 test_cnn.py --model_dir ./downloaded_pretrained_models/cropr224_Pcn --output_file ../shared/results/cropr224_Pcn_test.npz --crop_size 224 --base_network Pcn --list_dev_test 'D01;D02;D03;D04;D05;D06;D07;D08;D09;D10;D11;D12;D13;D14;D15;D16;D17;D18;D19;D20;D21;D22;D23;D24;D25;D26;D27;D28;D29;D30;D31;D32;D33;D34;D35'