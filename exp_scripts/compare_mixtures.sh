#!/bin/bash

expdir=1202_nmixtures_400K

python multitask_training.py --expname nmix200_400K_seed1 --n_mixtures 200 --device 0 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 5 --dataset 400K --seed 1 --log 2> ./errs/nmix200_400K.txt & 

python multitask_training.py --expname nmix400_400K_seed1 --n_mixtures 400 --device 0 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 5 --dataset 400K --seed 1 --log 2> ./errs/nmix400_400K.txt & 

python multitask_training.py --expname nmix800_400K_seed1 --n_mixtures 800 --device 0 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 5 --dataset 400K --seed 1 --log 2> ./errs/nmix800_400K.txt & 

python multitask_training.py --expname nmix1600_400K_seed1 --n_mixtures 1600 --device 2 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 5 --dataset 400K --seed 1 --log 2> ./errs/nmix1600_400K.txt & 

python multitask_training.py --expname nmix3200_400K_seed1 --n_mixtures 3200 --device 2 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 5 --dataset 400K --seed 1 --log 2> ./errs/nmix3200_400K.txt & 

python multitask_training.py --expname nmix6000_400K_seed1 --n_mixtures 6400 --device 3 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 5 --dataset 400K --seed 1 --log 2> ./errs/nmix6400_400K.txt & 

python multitask_training.py --expname nmix12800_400K_seed1 --n_mixtures 12800 --device 4 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 5 --dataset 400K --seed 1 --log 2> ./errs/nmix12800_400K.txt &

python multitask_training.py --expname nmix25600_400K_seed1 --n_mixtures 25600 --device 5 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 5 --dataset 400K --seed 1 --log 2> ./errs/nmix25600_400K.txt & 

python multitask_training.py --expname nmix51200_400K_seed1 --n_mixtures 51200 --device 6 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 5 --dataset 400K --seed 1 --log 2> ./errs/nmix51200_400K.txt 

wait

python multitask_training.py --expname nmix200_400K_seed2 --n_mixtures 200 --device 0 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 5 --dataset 400K --seed 2 --log 2> ./errs/nmix200_400K.txt & 

python multitask_training.py --expname nmix400_400K_seed2 --n_mixtures 400 --device 0 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 5 --dataset 400K --seed 2 --log 2> ./errs/nmix400_400K.txt & 

python multitask_training.py --expname nmix800_400K_seed2 --n_mixtures 800 --device 0 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 5 --dataset 400K --seed 2 --log 2> ./errs/nmix800_400K.txt & 

python multitask_training.py --expname nmix1600_400K_seed2 --n_mixtures 1600 --device 2 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 5 --dataset 400K --seed 2 --log 2> ./errs/nmix1600_400K.txt & 

python multitask_training.py --expname nmix3200_400K_seed2 --n_mixtures 3200 --device 2 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 5 --dataset 400K --seed 2 --log 2> ./errs/nmix3200_400K.txt & 

python multitask_training.py --expname nmix6000_400K_seed2 --n_mixtures 6400 --device 3 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 5 --dataset 400K --seed 2 --log 2> ./errs/nmix6400_400K.txt & 

python multitask_training.py --expname nmix12800_400K_seed2 --n_mixtures 12800 --device 4 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 5 --dataset 400K --seed 2 --log 2> ./errs/nmix12800_400K.txt &

python multitask_training.py --expname nmix25600_400K_seed2 --n_mixtures 25600 --device 5 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 5 --dataset 400K --seed 2 --log 2> ./errs/nmix25600_400K.txt & 

python multitask_training.py --expname nmix51200_400K_seed2 --n_mixtures 51200 --device 6 --expdir $expdir --train_ratio 1.0 --hidden_dim 128 --lr 0.000176 --weight_decay 0.0000117 --alpha 5 --dataset 400K --seed 2 --log 2> ./errs/nmix51200_400K.txt
