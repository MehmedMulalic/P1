#!/bin/bash

# Version Control
REQUIRED_VERSION="22.04"
CURRENT_VERSION=$(lsb_release -rs)

if dpkg --compare-versions "$CURRENT_VERSION" lt "$REQUIRED_VERSION"; then
    echo -e "\e[31mThis script requires Ubuntu $REQUIRED_VERSION."
    echo "You are running Ubuntu $CURRENT_VERSION.\e[0m"
    exit 1
fi

if ! lspci | grep -i 'nvidia' > /dev/null; then
    echo -e "\e[31mThis script requires an NVIDIA GPU.\e[0m"
    exit 1
fi

# Install Conda with packages
echo -e "\e[33mBeginning installation"
echo -e "Installing Miniforge3 with packages...\e[0m"
if command -v conda >/dev/null 2>&1; then
    echo -e "\e[33mMiniforge3 already installed, skipping...\e[0m"
else
    cd ~
    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3
    rm Miniforge3-Linux-x86_64.sh
    ~/miniforge3/bin/conda init
    echo 'export LIBRARY_PATH="$CONDA_PREFIX/lib:$LIBRARY_PATH"' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"' >> ~/.bashrc
    echo 'export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/lib/gcc/x86_64-conda-linux-gnu/14.2.0/include:$CONDA_PREFIX/lib/gcc/x86_64-conda-linux-gnu/14.2.0/include/c++/:$CONDA_PREFIX/lib/gcc/x86_64-conda-linux-gnu/14.2.0/include/c++/x86_64-conda-linux-gnu:$CPLUS_INCLUDE_PATH"' >> ~/.bashrc
    source ./.bashrc
fi

conda install -y -c conda-forge gcc=14.2.0 libcxx cmake boost gcc_linux-64 gxx_linux-64

# Install Mojo (installation could change in the future) [https://docs.modular.com/mojo/manual/get-started/]
echo -e "\e[33mInstalling Mojo...\e[0m"
if command -v pixi >/dev/null 2>&1; then
    echo -e "\e[33mMojo already installed, skipping...\e[0m"
else
    cd ~
    curl -fsSL https://pixi.sh/install.sh | sh
    echo 'export PATH="$HOME/.pixi/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
fi

# Install Cuda 12.8.1
echo -e "\e[33mInstalling Cuda...\e[0m"
if [ -d "$HOME/.local/cuda-12.8.1/" ]; then
    echo -e "\e[33mCUDA 12.8.1 already installed, skipping...\e[0m"
else
    cd ~
    wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
    chmod +x ./cuda_12.8.1_570.124.06_linux.run
    ./cuda_12.8.1_570.124.06_linux.run --toolkit --silent --override --toolkitpath=$HOME/.local/cuda-12.8.1
    rm ~/cuda_12.8.1_570.124.06_linux.run
    echo 'export LD_LIBRARY_PATH="$HOME/.local/cuda-12.8.1/lib64:$LD_LIBRARY_PATH"' >> ~/.bashrc
    echo 'export PATH=$HOME/cuda-12.8/bin:$PATH' >> ~/.bashrc
fi

# Install LLVM+Clang
echo -e "\e[33mInstalling LLVM+Clang...\e[0m"
if command -v clang >/dev/null 2>&1 && command -v llvm-config >/dev/null 2>&1; then
    echo -e "\e[33mLLVM+Clang already installed, skipping\e[0m"
else
    git clone --branch llvmorg-17.0.6 --single-branch https://github.com/llvm/llvm-project.git ~/llvm-project
    mkdir -p ~/llvm-project/build
    cd ~/llvm-project/build
    cmake -G "Unix Makefiles" ../llvm \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$HOME/.local/llvm \
    -DLLVM_ENABLE_PROJECTS="clang;lld;clang-tools-extra" \
    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_BUILD_LLVM_DYLIB=ON \
    -DCMAKE_CXX_FLAGS="-fext-numeric-literals"
    make -j$(nproc) -s
    make install -s
    cd ~
    rm -rf ~/llvm-project
    echo 'export PATH="$HOME/.local/llvm/bin:$PATH"' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH="$HOME/.local/llvm/lib:$LD_LIBRARY_PATH"' >> ~/.bashrc
    source ~/.bashrc
fi

# Install AdaptiveCpp
echo -e "\e[33mInstalling AdaptiveCpp...\e[0m"
if command -v acpp >/dev/null 2>&1; then
    echo -e "\e[33mAdaptiveCpp already installed, skipping\e[0m"
else
    cd ~
    git clone https://github.com/AdaptiveCpp/AdaptiveCpp.git
    mkdir -p ~/AdaptiveCpp/build
    cd ~/AdaptiveCpp/build
    cmake .. -G "Unix Makefiles" \
    -DCMAKE_VERBOSE_MAKEFILE=ON \
    -DCMAKE_MAKE_PROGRAM=make \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DWITH_CUDA_BACKEND=ON \
    -DCMAKE_CXX_STANDARD_REQUIRED=ON \
    -DCMAKE_INSTALL_PREFIX=$HOME/.local/adaptivecpp \
    -DCUDA_TOOLKIT_ROOT_DIR=$HOME/.local/cuda-12.8.1 \
    -DCMAKE_CXX_COMPILER=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++ \
    -DCMAKE_C_COMPILER=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc \
    -DCMAKE_CXX_FLAGS="-I$CONDA_PREFIX/lib/gcc/x86_64-conda-linux-gnu/14.2.0/include/c++ \
    -I$CONDA_PREFIX/lib/gcc/x86_64-conda-linux-gnu/14.2.0/include/c++/x86_64-conda-linux-gnu \
    -I$CONDA_PREFIX/lib/gcc/x86_64-conda-linux-gnu/14.2.0/include"
    make -j$(nproc)
    make install
    cd ~
    rm -rf ~/AdaptiveCpp/
    echo 'export PATH="$HOME/.local/adaptivecpp/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
fi

echo -e "\e[33mFinished installation\e[0m"

# OpenCL NVIDIA vendor check
echo -e "\e[33mChecking for vendors...\e[0m"
if [ -e "/etc/OpenCL/vendors/nvidia.icd" ]; then
    echo "export OPENCL_VENDOR_PATH=/etc/OpenCL/vendors" >> ~/.bashrc
    echo -e "\e[32mVendor defined\e[0m"
    source ~/.bashrc
else
    mkdir ~/opencl-vendors
    echo "$HOME/cuda-12.8.1/lib64/libOpenCL.so" > "$HOME/opencl-vendors/nvidia.icd"
    echo "export OPENCL_VENDOR_PATH=$HOME/opencl-vendors" >> ~/.bashrc
    echo -e "\e[32mVendor defined through CUDA\e[0m"
    source ~/.bashrc
fi

# Installation check
if command -v conda >/dev/null 2>&1; then
    echo -e "\e[32mMiniforge3 SUCCESSFULLY installed\e[0m"
else
    echo -e "\e[31mMiniforge3 FAILED to install\e[0m"
fi

if command -v pixi >/dev/null 2>&1; then
    echo -e "\e[32mMojo SUCCESSFULLY installed\e[0m"
else
    echo -e "\e[31mMojo FAILED to install\e[0m"
fi

if [ -d "$HOME/.local/cuda-12.8.1/" ]; then
    echo -e "\e[32mCUDA 12.8.1 SUCCESSFULLY installed\e[0m"
else
    echo -e "\e[31mCUDA 12.8.1 FAILED to install\e[0m"
fi

if command -v clang >/dev/null 2>&1 && command -v llvm-config >/dev/null 2>&1; then
    echo -e "\e[32mLLVM+Clang SUCCESSFULLY installed\e[0m"
else
    echo -e "\e[31mLLVM+Clang FAILED to install\e[0m"
fi

if command -v acpp >/dev/null 2>&1; then
    echo -e "\e[32mAdaptiveCpp SUCCESSFULLY installed\e[0m"
else
    echo -e "\e[31mAdaptiveCpp FAILED to install\e[0m"
fi

# Building the project
echo -e "\e[33mBuilding files...\e[0m"
cd ~/P1
mkdir build && cd build
x86_64-conda-linux-gnu-g++ ../src/Sequential/sequential.cpp -O3 -o ./sequential
x86_64-conda-linux-gnu-g++ -I$HOME/.local/cuda-12.8.1/targets/x86_64-linux/include -L$HOME/.local/cuda-12.8.1/lib64 -lOpenCL ../src/OpenCL/opencl.cpp -O3 -o ./opencl
acpp ../src/SYCL/sycl.cpp -O3 -o ./sycl
acpp ../src/SYCL/sycp.cpp -O3 -DBUFFER -o ./sycl_buffer
pixi init mojo -c https://conda.modular.com/max-nightly/ -c conda-forge
cd mojo
pixi add modular
pixi run mojo build $HOME/P1/src/Mojo/MojoGPU.mojo
