#!/bin/bash

expdir=1203_multitask

python multitask_training.py --expname alpha0_seed1 --n_mixtures 51200 --device 3 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 0 --dataset 400K --seed 1 --log &

python multitask_training.py --expname alpha1_seed1 --n_mixtures 51200 --device 4 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 1 --dataset 400K --seed 1 --log &

python multitask_training.py --expname alpha2_seed1 --n_mixtures 51200 --device 5 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 2 --dataset 400K --seed 1 --log &

python multitask_training.py --expname alpha5_seed1 --n_mixtures 51200 --device 6 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 5 --dataset 400K --seed 1 --log &

python multitask_training.py --expname alpha10_seed1 --n_mixtures 51200 --device 7 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 10 --dataset 400K --seed 1 --log

wait

python multitask_training.py --expname alpha0_seed2 --n_mixtures 51200 --device 3 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 0 --dataset 400K --seed 2 --log &

python multitask_training.py --expname alpha1_seed2 --n_mixtures 51200 --device 4 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 1 --dataset 400K --seed 2 --log &

python multitask_training.py --expname alpha2_seed2 --n_mixtures 51200 --device 5 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 2 --dataset 400K --seed 2 --log &

python multitask_training.py --expname alpha5_seed2 --n_mixtures 51200 --device 6 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 5 --dataset 400K --seed 2 --log &

python multitask_training.py --expname alpha10_seed2 --n_mixtures 51200 --device 7 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 10 --dataset 400K --seed 2 --log