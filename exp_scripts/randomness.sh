#!/bin/bash

expdir=1209_randomness

python multitask_training.py --expname seed0 --n_mixtures 51200 --device 3 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 5 --dataset 400K --seed 0 --log &

python multitask_training.py --expname seed1 --n_mixtures 51200 --device 4 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 5 --dataset 400K --seed 1 --log &

python multitask_training.py --expname seed2 --n_mixtures 51200 --device 5 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 5 --dataset 400K --seed 2 --log &

python multitask_training.py --expname seed3 --n_mixtures 51200 --device 6 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 5 --dataset 400K --seed 3 --log &

python multitask_training.py --expname seed4 --n_mixtures 51200 --device 7 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 5 --dataset 400K --seed 4 --log