export CUDA_VISIBLE_DEVICES=0,1

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=2

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=log/Qwen_megatron
TENSORBOARD_LOGS_PATH=log/Qwen_megatron #<Specify path>
VOCAB_FILE=gpt2_vocab/vocab.json #<Specify path to file>/gpt2-vocab.json
MERGE_FILE=gpt2_vocab/merges.txt #<Specify path to file>/gpt2-merges.txt
#<Specify path and file prefix>_text_document
TRAIN_PATH=''
for i in {1..99}; do
  TRAIN_PATH+=" 1.0 edu_fineweb10B_megatron/$(printf "%d" $i)_text_document"
done

VAL_PATH='
1.0
edu_fineweb10B_megatron/100_text_document'

# DATA_PATH='\
# 1.0
# edu_fineweb10B_megatron/1_text_document'

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --num-layers 10 
    --hidden-size 768 
    --ffn-hidden-size 3072 
    --num-attention-heads 12 
    --attention-backend auto # Can use (flash/fused/unfused/local)
    --use-flash-attn 
    --add-qkv-bias 
    --disable-bias-linear 
    --seq-length 1024 
    --group-query-attention 
    --num-query-groups 2 
    --max-position-embeddings 1024 
    --position-embedding-type rope 
    --rotary-base 10000
    --no-rope-fusion 
    --normalization RMSNorm 
    --norm-epsilon 1e-6 
    --swiglu 
    #--untie-embeddings-and-output-weights
)

REGULARIZATION_ARGS=(
    --attention-dropout 0 
    --hidden-dropout 0 
    --weight-decay 0.1 
    --clip-grad 1.0 
)

TRAINING_ARGS=(
    --micro-batch-size 32 
    --global-batch-size 512 
    --train-iters 20000 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --adam-eps 1e-8 
    --init-method-std 0.02 
    --seed 1337 
    --bf16
    --lr 1.0e-3 
    --lr-decay-style cosine 
    --min-lr 1.0e-4
    --lr-warmup-iters 715
    --lr-decay-iters 19200 
    --no-gradient-accumulation-fusion 
    --no-masked-softmax-fusion 
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 2 
	--pipeline-model-parallel-size 1 
    --transformer-impl local
)

TOKENIZER_ARGS=(
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --vocab-size 50304
    --tokenizer-type GPT2BPETokenizer 
)

DATA_ARGS=(
    --train-data-path $TRAIN_PATH 
    --valid-data-path $VAL_PATH 
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 2500 
    --eval-interval 250 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_megatron.py \
    ${MODEL_ARGS[@]} \
    ${REGULARIZATION_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
