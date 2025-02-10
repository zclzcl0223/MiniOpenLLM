import re
import torch
import os
import json
import transformers
import argparse
import warnings
warnings.filterwarnings('ignore')
from datasets import load_dataset, Dataset
from model.Qwen.Qwen import Qwen
from types import SimpleNamespace
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from trl import DataCollatorForCompletionOnlyLM

# Load & prepare dataset
SYSTEM_PROMPT = """Answer the following question step by step:"""

response_template = "<|im_start|>assistant\n"

def extract_answer(text):
    # the answer is marked by '####'
    if '####' not in text:
        return None
    return text.split('####')[1].strip()

def extract_xml_answer(text):
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def get_gsm8k(split="train"):
    # turn dataset from conversation (sft) to question-answer (rl)
    data = load_dataset("openai/gsm8k", 'main')
    data = data.map(lambda x: {
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']},
            {'role': 'assistant', 'content': x['answer']}
        ],
    }, remove_columns=["question", "answer"])
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_from_checkpoint", action="store_true")
    parser.add_argument("--use_compile", action="store_true")
    parser.add_argument("--model_type", type=str, default="Qwen")
    parser.add_argument("--config_path", type=str, default="grpo_config_0.5B")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "infer"])
    args = parser.parse_args()

    config_path = f"model/{args.model_type}/{args.config_path}.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    config = SimpleNamespace(**config)

    ddp = int(os.environ.get('RANK', -1)) != -1 # ddp run

    if ddp:
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE']) # num of gpu
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        print(f"using device: {device}")
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    torch.manual_seed(114514)
    torch.cuda.manual_seed(114514)
    torch.set_float32_matmul_precision('high')
    temperature = 0.7
    top_p = 0.8
    top_k = 20

    dataset = get_gsm8k()

    output_dir = "outputs/Qwen2.5-0.5B-SFT"
    run_name = "Qwen2.5-0.5B-SFT"

    training_args = SFTConfig(
        output_dir=output_dir,
        run_name=run_name,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        logging_steps=1,
        logging_strategy="steps",
        bf16=True,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        max_seq_length=1024,
        num_train_epochs=100,
        save_steps=2500,
        max_grad_norm=0.1,
        eval_strategy="steps",
        eval_steps=50,
        report_to=None,
        log_on_each_node=False,
        local_rank=ddp_local_rank,
        label_smoothing_factor=0.01
    )

    model_name = "Qwen/Qwen2.5-0.5B"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    #model = Qwen.from_pretrained(config, model_name, ddp_rank)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    if args.use_compile:
        if master_process:
            print(f"compiling model...")
        model = torch.compile(model)
    # only compute loss on the response
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=collator
    )
    trainer.train()

    if ddp:
        destroy_process_group()
