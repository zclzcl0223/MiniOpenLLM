export CUDA_VISIBLE_DEVICES=4,5,6,7

# with ddp
#torchrun --standalone --nproc_per_node=4 pipeline.py --model_type GPT --config_path config_124M --use_compile
#torchrun --standalone --nproc_per_node=4 pipeline.py --model_type GPT --config_path config_124M --use_compile --mode infer --resume_from_checkpoint

#torchrun --standalone --nproc_per_node=2 pipeline.py --model_type Qwen --config_path config_123M --use_compile
#torchrun --standalone --nproc_per_node=2 pipeline.py --model_type Qwen --config_path config_140M --use_compile

torchrun --standalone --nproc_per_node=4 pipeline.py --model_type DeepSeek --config_path config_123M --use_compile
torchrun --standalone --nproc_per_node=4 pipeline.py --model_type DeepSeek --config_path config_126M --use_compile

# without ddp
#python pipeline.py --model_type DeepSeek --config_path config_126M --use_compile
