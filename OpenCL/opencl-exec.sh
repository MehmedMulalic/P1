#!/bin/bash
echo "OpenCL execution started"

printf "\n### NEW TEST ITERATION ###\n\n" >> opencl-out.log
for i in 10 100; do
    for j in 10 100 1000; do
        ./opencl $j $i >> opencl-out.log 2>&1
    done
done
./opencl 10 1000 >> opencl-out.log 2>&1

echo "OpenCL execution completed"