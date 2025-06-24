#!/bin/bash
echo "SYCL build started"

for i in 10 100 1000; do
    for j in 10 100 1000; do
        ./sycl $j $i >> sycl-out.log 2>&1
    done
done

echo "SYCL build completed"