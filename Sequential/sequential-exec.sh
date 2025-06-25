#!/bin/bash
echo "Sequential execution started"

printf "\n### NEW TEST ITERATION ###\n" >> sequential-out.log
for i in 10 100; do
    for j in 10 100 1000; do
        ./sequential-cpp $j $i >> sequential-out.log 2>&1
    done
done
./sequential-cpp 10 1000 >> sequential-out.log 2>&1
./sequential-cpp 1000 200 >> sequential-out.log 2>&1

echo "Sequential execution completed"