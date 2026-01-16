#!/bin/bash

# Initialize conda (adjust path if needed)
source ~/anaconda3/etc/profile.d/conda.sh
conda activate bl-stssdl

# Change to ST-SSDL directory (where PEMS data folders are located)
cd /home/cds/ltkien/STLLM-baseline/ST-SSDL

python generate_training_data_his_PEMS.py --dataset PEMS04
python generate_training_data_his_PEMS.py --dataset PEMS07
python generate_training_data_his_PEMS.py --dataset PEMS08

cd model_STSSDL

python train_STSSDL.py --gpu 0 --dataset PEMS04
python train_STSSDL.py --gpu 0 --dataset PEMS07
python train_STSSDL.py --gpu 0 --dataset PEMS08