conda activate bl-stssdl

python generate_training_data_his_PEMS.py --dataset PEMS04
python generate_training_data_his_PEMS.py --dataset PEMS07
python generate_training_data_his_PEMS.py --dataset PEMS08

cd model_STSSDL

python train_STSSDL.py --gpu 0 --dataset PEMS04
python train_STSSDL.py --gpu 0 --dataset PEMS07
python train_STSSDL.py --gpu 0 --dataset PEMS08