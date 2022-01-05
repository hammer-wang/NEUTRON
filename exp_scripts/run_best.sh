#!/bin/bash

gpu=0
expdir=log

python multitask_training.py --expname best --n_mixtures 1024 --device $gpu --expdir $expdir --train_ratio 1.0 --hidden_dim 512 --lr 0.0001 --weight_decay 0.00001 --alpha 1 --log