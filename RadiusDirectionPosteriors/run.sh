#!/bin/bash

echo Running on $HOSTNAME
module load anaconda/3
conda activate PGM

lr=$1
bs=$2
optim=$3
data=$4
model=$5

python -u main.py --lr $lr --bs $bs --optim $optim \
--data $data --model_type $model --prior_file HalfCauchy-fc0.01-conv0.01.json --epochs 100