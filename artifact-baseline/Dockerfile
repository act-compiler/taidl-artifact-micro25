FROM amd64/ubuntu:jammy

# Base dependencies for Gemmini (Chipyard), Intel SDE, and C compilation
RUN apt-get update \
  && apt-get install -y sudo wget cmake make git clang-14 curl file apt-transport-https build-essential cpio autoconf g++ device-tree-compiler \
  && apt-get clean -y \
  && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py311_24.11.1-0-Linux-x86_64.sh -O miniconda.sh
RUN chmod +x miniconda.sh && \
    ./miniconda.sh -b -u -p /opt/miniconda && \
    rm miniconda.sh && \
    /opt/miniconda/bin/conda init bash

COPY requirements.yml /tmp/requirements.yml

ENV CONDA_PLUGINS_AUTO_ACCEPT_TOS="yes"
RUN /opt/miniconda/bin/conda env create -f /tmp/requirements.yml && \
    /opt/miniconda/bin/conda clean -afy

WORKDIR /workspace

ENV PATH=/opt/miniconda/envs/micro2025-baseline/bin:$PATH

# Setup Intel SDE for AMX
WORKDIR /amx/
RUN INTEL_SDE_URL=$(curl -s https://www.intel.com/content/www/us/en/download/684897/intel-software-development-emulator.html | grep -Po 'https://downloadmirror.intel.com/.*lin.tar.xz(?=")') && \
    wget -qO intel-sde.tar.xz $INTEL_SDE_URL && \
    tar xf intel-sde.tar.xz --strip-components=1

# Setup Gemmini
WORKDIR /gemmini
RUN git clone https://github.com/ucb-bar/chipyard.git
WORKDIR /gemmini/chipyard
RUN git checkout 1.9.1
RUN ./build-setup.sh riscv-tools -s 2 -s 4 -s 5 -s 6 -s 7 -s 8 -s 9 -s 10
ENV RISCV=/gemmini/chipyard/.conda-env/riscv-tools
WORKDIR /gemmini/chipyard/generators/gemmini
RUN git config remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*" && \
    git fetch && git checkout v0.7.1 && \
    git submodule update --init --recursive


WORKDIR /workspace

CMD ["bash"]
