#!/bin/bash
echo "Mojo execution started"

printf "\n### NEW TEST ITERATION ###\n\n" >> ./mojo-out.log
for i in 10 100; do
    for j in 10 100 1000; do
        ./MojoGPU $j $i >> ./mojo-out.log 2>&1
    done
done

./MojoGPU 10 1000 >> ./mojo-out.log 2>&1
./MojoGPU 1000 200 >> ./mojo-out.log 2>&1
./MojoGPU 1000 500 >> ./mojo-out.log 2>&1
./MojoGPU 1000 800 >> ./mojo-out.log 2>&1

echo "Mojo execution completed"