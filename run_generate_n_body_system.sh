#!/bin/sh

for i in 50 100
    do
        python -u n_body_system/dataset/generate_dataset.py --num-train=10000 --seed=43 --sufix=charged-$i --n_balls=$i > log/generate-charged-$i 2>&1
    done

