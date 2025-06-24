#!/bin/bash
x86_64-conda-linux-gnu-g++ -I$HOME/cuda-12.8/targets/x86_64-linux/include -L$HOME/cuda-12.8/lib64 -lOpenCL "$@"