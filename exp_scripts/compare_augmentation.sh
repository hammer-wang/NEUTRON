#!/bin/bash

expdir=1206_compare_augmentation

python multitask_training.py --expname primitive_100K_seed3 --n_mixtures 51200 --device 0 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 5 --dataset primitive_100K --epochs 500 --log --seed 3 &

python multitask_training.py --expname primitive_100K_seed4 --n_mixtures 51200 --device 1 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 5 --dataset primitive_100K --epochs 500 --log --seed 4 &

python multitask_training.py --expname primitive_150K_seed3 --n_mixtures 51200 --device 2 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 5 --dataset primitive_150K --epochs 500 --log --seed 3 &

python multitask_training.py --expname primitive_150K_seed4 --n_mixtures 51200 --device 3 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 5 --dataset primitive_150K --epochs 500 --log --seed 4 &

python multitask_training.py --expname augmented_150K_seed3 --n_mixtures 51200 --device 4 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 5 --dataset augmented_150K --epochs 500  --log --seed 3 &

python multitask_training.py --expname augmented_150K_seed4 --n_mixtures 51200 --device 5 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 5 --dataset augmented_150K --epochs 500  --log --seed 4 &

python multitask_training.py --expname augmented_400K_seed3 --n_mixtures 51200 --device 6 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 5 --dataset 400K --epochs 500  --log --seed 3 &

python multitask_training.py --expname augmented_400K_seed4 --n_mixtures 51200 --device 7 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 5 --dataset 400K --epochs 500  --log --seed 4