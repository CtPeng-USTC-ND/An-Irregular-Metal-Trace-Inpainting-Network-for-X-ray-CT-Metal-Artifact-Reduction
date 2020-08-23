#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/

cd src/cuda
echo "Compiling my_lib kernels by nvcc..."
/usr/local/cuda-8.0/bin/nvcc -c -o fan_projection_kernel.cu.o fan_projection_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52

cd ../../

python build.py