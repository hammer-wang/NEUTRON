#!/bin/bash

gpu=6
expdir=1124_nmixtures

# python multitask_training.py --log --expname nmix16 --n_mixtures 16 --device $gpu --expdir $expdir &
# python multitask_training.py --log --expname nmix32 --n_mixtures 32 --device $gpu --expdir $expdir &
# python multitask_training.py --log --expname nmix64 --n_mixtures 64 --device $gpu --expdir $expdir &
# python multitask_training.py --log --expname nmix128 --n_mixtures 128 --device $gpu --expdir $expdir & 
# python multitask_training.py --log --expname nmix256 --n_mixtures 256 --device $gpu --expdir $expdir & 
# python multitask_training.py --log --expname nmix512 --n_mixtures 512 --device $gpu --expdir $expdir & 
# python multitask_training.py --log --expname nmix1024 --n_mixtures 1024 --device $gpu --expdir $expdir &
# python multitask_training.py --log --expname nmix2048 --n_mixtures 2048 --device $gpu --expdir $expdir 

# python multitask_training.py --log --expname nmix8192 --n_mixtures 8192 --device 2 --expdir $expdir 

python multitask_training.py --log --expname nmix4096 --n_mixtures 4096 --device 0 --expdir $expdir &
python multitask_training.py --log --expname nmix16384 --n_mixtures 16384 --device 1 --expdir $expdir
