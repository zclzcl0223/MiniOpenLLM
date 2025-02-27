if [ ! -d "./edu_fineweb10B_megatron" ]; then
    mkdir ./edu_fineweb10B_megatron
fi

for i in {01..99}
do
python megatron_data_convert.py \
    --input edu_fineweb10B/edufineweb_train_0000${i}.npy \
    --output edu_fineweb10B_megatron/$(printf "%d" $((10#$i))) \
    --vocab-file gpt2_vocab/vocab.json \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file gpt2_vocab/merges.txt \
    --workers 1
done

python megatron_data_convert.py \
    --input edu_fineweb10B/edufineweb_val_000000.npy \
    --output edu_fineweb10B_megatron/100 \
    --vocab-file gpt2_vocab/vocab.json \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file gpt2_vocab/merges.txt \
    --workers 1
