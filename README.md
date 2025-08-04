# Praktikum P1 project for University of Vienna Faculty of Computer Science

This project aims to showcase the difference in performance between three different heterogeneous parallel framworks: Mojo, SYCL, and OpenCL.  
The project was conducted on the University's UCS9 cluster of 6 nodes. The cluster runs on Ubuntu 22.04 with an AMD EPYC 7543 32-core CPU and an NVIDIA Tesla A100 GPU. The files are built using GCC version 14.2.0 with CUDA toolkit version 12.8.1_570.124.06 and LLVM version 17.0.6 including Clang. 

---

## Installation

After cloning the repository, run the install script which installs all requirements and builds the project.

```bash
git clone https://github.com/MehmedMulalic/P1.git ~/P1
chmod +x ~/P1/install.sh
./install.sh
```

### Requirements
* Miniforge3
    * CMake
    * Boost
    * gcc, gxx, and libcxx
* Pixi
* CUDA toolkit 12.8.1
* LLVM 17.0.6 with Clang
* AdaptiveCPP (formerly hipSYCL) with CXX standard 17

---

## Usage

There are five executables which have the same argument parsing usage:  
Usage (positive particle_count, positive grid_size): <particle_count>, <grid_size>

The default values for atom count and grid mesh size is (1, 1) but they can be changed if parsed with the described usage.
