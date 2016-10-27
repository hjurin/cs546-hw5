#!/bin/bash

cuda_progs="cuda1 cuda2 cuda3"
for matrix_size in `seq 20 7980 8000`;
do
    echo -e "Computing with serial\t[matrix_size = $matrix_size]"
    ./serial $matrix_size 0 > /dev/null
    for program in $cuda_progs
    do
        for block_size in `seq 1 1 1024`;
        do
            echo -e "Computing with $program\t[block_size = $block_size] [matrix_size = $matrix_size]"
            ./$program $matrix_size 0 $block_size > /dev/null
        done
    done
done
