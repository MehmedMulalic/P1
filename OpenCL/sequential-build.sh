#!/bin/bash
echo "Sequential build started"

echo "### NEW TEST ITERATION ###" >> sequential-out.log
for i in 10 100; do
    for j in 10 100 1000; do
        ./sequential-CPP $j $i >> sequential-out.log 2>&1
    done
done
./sequential-CPP 10 1000 >> sequential-out.log 2>&1

echo "Sequential build completed"