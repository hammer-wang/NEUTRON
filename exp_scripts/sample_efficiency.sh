#!/bin/bash

expdir=1203_sample_efficiency

python multitask_training.py --expname ratio0.01 --n_mixtures 51200 --device 2 --expdir $expdir --train_ratio 0.01 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 5 --dataset 400K --seed 42 --log &

python multitask_training.py --expname ratio0.05 --n_mixtures 51200 --device 3 --expdir $expdir --train_ratio 0.05 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 5 --dataset 400K --seed 42 --log &

python multitask_training.py --expname ratio0.1 --n_mixtures 51200 --device 4 --expdir $expdir --train_ratio 0.1 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 5 --dataset 400K --seed 42 --log &

python multitask_training.py --expname ratio0.25 --n_mixtures 51200 --device 5 --expdir $expdir --train_ratio 0.25 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 5 --dataset 400K --seed 42 --log &

python multitask_training.py --expname ratio0.5 --n_mixtures 51200 --device 6 --expdir $expdir --train_ratio 0.5 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 5 --dataset 400K --seed 42 --log &

python multitask_training.py --expname ratio1.0 --n_mixtures 51200 --device 7 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 5 --dataset 400K --seed 42 --log