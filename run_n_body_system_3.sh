#!/bin/sh

dataset=_charged20_initvel1charged-20
model=egnn_vel
for i in $(seq 1 20)
    do
        python -u main_nbody.py --exp_name=exp-20-egnn_vel-$i --model=$model --dataset=$dataset --max_training_samples=3000 --lr=5e-4 --n_layers=$i --gpu=3 > log/$model-$dataset-$i 2>&1
    done

