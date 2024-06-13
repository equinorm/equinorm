#!/bin/sh

dataset=_charged5_initvel1charged-5
model=egnn_vel
for i in $(seq 1 20)
    do
        python -u main_nbody.py --exp_name=exp-5-egnn_vel-$i --model=$model --dataset=$dataset --max_training_samples=3000 --lr=5e-4 --n_layers=$i --gpu=1 > log/$model-$dataset-$i 2>&1
    done

