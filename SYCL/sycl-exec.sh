#!/bin/bash
echo "SYCL execution started"

printf "\n### NEW TEST ITERATION ###\n\n" >> ./sycl-out.log
for i in 10 100; do
    for j in 10 100 1000; do
        ./sycl $j $i >> ./sycl-out.log 2>&1
    done
done

./sycl 10 1000 >> ./sycl-out.log 2>&1
./sycl 1000 200 >> ./sycl-out.log 2>&1
./sycl 1000 500 >> ./sycl-out.log 2>&1
./sycl 1000 800 >> ./sycl-out.log 2>&1

for i in 10 100; do
    for j in 10 100 1000; do
        ./sycl $j $i buffer >> ./sycl-out.log 2>&1
    done
done

./sycl 10 1000 buffer >> ./sycl-out.log 2>&1
./sycl 1000 200 buffer >> ./sycl-out.log 2>&1
./sycl 1000 500 buffer >> ./sycl-out.log 2>&1
./sycl 1000 800 buffer >> ./sycl-out.log 2>&1

echo "SYCL execution completed"