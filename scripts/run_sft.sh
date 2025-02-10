export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

torchrun --standalone --nproc_per_node=6 post_trainer_sft.py | tee -a log_sft.txt
