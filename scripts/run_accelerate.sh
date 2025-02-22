export CUDA_VISIBLE_DEVICES=0,1

accelerate launch --num_processes=2 pretrain_accelerate.py --model_type GPT --config_path config_124M --mode train
