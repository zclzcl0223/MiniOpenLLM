import re
import torch
import os
import json
import transformers
import argparse
from datasets import load_dataset, Dataset
from model.Qwen.Qwen import Qwen
from types import SimpleNamespace
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# reference: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb

# Load & prepare dataset
SYSTEM_PROMPT = """
Respond in the following format:

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

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
    data = load_dataset("openai/gsm8k", 'main')[split]
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_answer(x['answer'])
    })
    return data

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs):
    # reward when getting right answers
    # prompts: (n, ), str
    # completions: (n, ), str
    # answer: (n, ), str
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(response) for response in responses]
    # visualize the first question
    q = prompts[0][-1]['content']
    #print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}",
    #      f"\nResponse:{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def number_reward_func(completions, **kwargs):
    # the answer is supposed to be number
    # completions: (n, ), str
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(response) for response in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs):
    # the response should include <reasoning> and <answering>
    # completions: (n, ), str
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]['content'] for completion in completions]
    mathces = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in mathces]

def soft_format_reward_func(completions, **kwargs):
    # completions: (n, ), str
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses] 
    return [0.5 if match else 0.0 for match in matches]

def xmlcount_reward_func(completions, **kwargs):
    # reward when responses include some right format
    def count_xml(text):
        count = 0.0
        if text.count("<reasoning>\n") == 1:
            count += 0.125
        if text.count("\n</reasoning>\n") == 1:
            count += 0.125
        if text.count("\n<answer>\n") == 1:
            count += 0.125
            # nothing is expected after the answer
            count -= len(text.split("\n</answer>\n")[-1])*0.001
        if text.count("\n</answer>") == 1:
            count += 0.125
            # nothing is expected after the answer (-1 means '\n' is not considered)
            count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
        return count
    responses = [completion[0]['content'] for completion in completions]
    return [count_xml(r) for r in responses]

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

    output_dir = "outputs/Qwen2.5-0.5B-GRPO"
    run_name = "Qwen2.5-0.5B-GRPO-gsm8k"

    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        learning_rate=1e-5,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        logging_steps=1,
        logging_strategy="steps",
        bf16=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_generations=16,
        max_prompt_length=256,
        max_completion_length=256,
        num_train_epochs=1,
        save_strategy="no",
        save_steps=100,
        max_grad_norm=0.1,
        report_to=None,
        use_vllm=False,
        vllm_gpu_memory_utilization=.3,
        vllm_device=device,
        log_on_each_node=False,
        local_rank=ddp_local_rank
    )

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    #model = Qwen.from_pretrained(config, model_name, ddp_rank)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    if args.use_compile:
        if master_process:
            print(f"compiling model...")
        model = torch.compile(model)
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            correctness_reward_func,
            number_reward_func,
            strict_format_reward_func,
            soft_format_reward_func,
            #xmlcount_reward_func
        ],
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

    if ddp:
        destroy_process_group()
    """
    hf_state_dict = model.state_dict()

    prompt = "Hello! I am a language assistant."
    token = tokenizer(prompt, return_tensors="pt").to(device)
    print(token)
    x, mask = token['input_ids'], token['attention_mask']
    #print(model(x, mask)[0].shape)

    max_length = 30

    while x.shape[1] < max_length:
        with torch.no_grad():
            with torch.autocast(device_type, dtype=torch.bfloat16):
                # (B, T, vocab) -> (B, 1, vocab)
                logits = model(x)[0][:, -1, :]
                probs = F.softmax(logits/temperature, dim=-1)
                topk_prob, topk_indices = torch.topk(probs, top_k, dim=-1)
                ix = torch.multinomial(topk_prob, 1)
                xcol = torch.gather(topk_indices, -1, ix)
                x = torch.cat([x, xcol], dim=-1)

    print(tokenizer.batch_decode(x))
    """
