import math
import torch
import inspect
import tiktoken
import time
import os
import json
import logging
import warnings
warnings.filterwarnings('ignore')
import torch.distributed as dist
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from dataclasses import dataclass
from model.GPT.GPT import GPT
from model.DeepSeek.DeepSeek import DeepSeek
from model.Qwen.Qwen import Qwen
from dataloader_torch import TokenDataset, get_dataloader
from accelerate import Accelerator, DeepSpeedPlugin, load_checkpoint_in_model
from accelerate.utils import ProjectConfiguration
from tqdm import trange
from tqdm.auto import tqdm
from accelerate.logging import get_logger

def unwrap_model(model):
    # Unwrap DDP
    if hasattr(model, 'module'):
        model = model.module
    # Unwrap torch.compile
    if hasattr(model, '_orig_mod'):
        model = model._orig_mod
    return model

def get_lr_lambda(it, warmup_steps, max_steps, max_lr, min_lr):
    # warm up
    if it < warmup_steps:
        return it / warmup_steps
    # after cosine decay
    if it > max_steps:
        return min_lr / max_lr
    # cosine decay
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return (min_lr + coeff * (max_lr - min_lr)) / max_lr

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

def main(model, config):
    
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

    # log
    log_dir = f"log/{config.model_type}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")

    # seeds
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # gradient accumulation
    total_batch_size = config.token_per_epoch # 2**19, ~0.5M, the number of tokens per batch
    B = config.batch_size
    T = config.block_size
    assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisble by B * T"
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

    # deepspeed
    deepspeed = DeepSpeedPlugin(zero_stage=2, 
                                    gradient_clipping=1.0, 
                                    gradient_accumulation_steps=grad_accum_steps,
                                    offload_optimizer_device="none",
                                    offload_param_device="none",
                                )
    accelerator_project_config = ProjectConfiguration(automatic_checkpoint_naming=True)
    accelerator = Accelerator(deepspeed_plugin=deepspeed, 
                              project_dir=log_dir,
                              project_config=accelerator_project_config,
                              mixed_precision="bf16",
                              )
    if accelerator.is_main_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
    torch.set_float32_matmul_precision('high')

    # dataset
    train_loader = get_dataloader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
    val_loader = get_dataloader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')

    # lr
    max_lr = config.max_lr
    min_lr = max_lr * 0.1
    warmup_steps = config.warmup_steps  # 375M / 2**19
    max_steps = config.max_steps  # 10B / 2**19
    total_steps = config.total_steps

    # real vocab_size: 50257
    model = model(config, process_rank=ddp_rank)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    log_file = os.path.join(log_dir, f"log_{total_params:.0f}M.txt")
    if accelerator.is_main_process:
        print(f"Parameters: {total_params:.0f}M")

    use_compile = config.use_compile
    if use_compile:
        if accelerator.is_main_process:
            print(f"compiling model...")
        model = torch.compile(model)

    if config.mode == "train":
        # with weight decay
        optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=config.max_lr, device_type=device_type)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda it: get_lr_lambda(it, warmup_steps, max_steps, max_lr, min_lr))
        model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
        accelerator.register_for_checkpointing(scheduler)
        
        if config.resume_from_checkpoint:
            checkpoint_files = [f for f in os.listdir(os.path.join(log_dir, "checkpoints")) if f.startswith("checkpoint")]
            assert len(checkpoint_files) > 0, "no checkpoints found"
            latest_checkpoint = sorted(checkpoint_files)[-1] # load the latest checkpoint
            checkpoint_path = os.path.join(log_dir, "checkpoints", latest_checkpoint)
            accelerator.load_state(checkpoint_path)
            global_step = scheduler.last_epoch
            if accelerator.is_main_process:
                print(f"resuming training from step {global_step}")
        else:
            global_step = 0
            with open(log_file, "w") as f: # clear log
                pass
    
        progress_bar = tqdm(range(global_step, total_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        def evaluate():
            model.eval()
            # eval on valiation set
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = int(640 / B)
                for i, batch in enumerate(val_loader):
                    x, y = batch
                    with accelerator.autocast():
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += accelerator.reduce(loss.detach(), reduction='mean')
                    if i == val_loss_steps - 1:
                        break
            if accelerator.is_main_process:
                tqdm.write(f"validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{global_step} val {val_loss_accum.item():.4f}\n")

            # save checkpoint
            if global_step > 0 and (global_step % 2500 == 0 or global_step == total_steps - 1):
                accelerator.save_state()
                # save the final model
                accelerator.save_model(model, max_shard_size='500MB', save_directory=log_dir)

            # eval on hellaswag
            num_correct_norm = 0
            num_total = 0
            for i, example in enumerate(iterate_examples("val")):
                if i % ddp_world_size != ddp_rank:
                    continue
                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(accelerator.device)
                mask = mask.to(accelerator.device)
                with torch.no_grad():
                    with accelerator.autocast():
                        logits, loss = model(tokens)
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm==label)
            if ddp:
                num_total = torch.tensor(num_total, dtype=torch.long, device=accelerator.device)
                num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=accelerator.device)
                num_total = accelerator.reduce(num_total, reduction='sum').item()
                num_correct_norm = accelerator.reduce(num_correct_norm, reduction='sum').item()
            acc_norm = num_correct_norm / num_total
            if accelerator.is_main_process:
                tqdm.write(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{global_step} hella {acc_norm:.4f}\n")

        # eval at first step
        if not config.resume_from_checkpoint:
            evaluate()
        t0 = time.time()
        loss_accum = 0.0
        gradient_accumulation_step = 0
        while global_step < total_steps:
            # gradient accumulation
            for i, batch in enumerate(train_loader):
                model.train()
                x, y = batch
                with accelerator.autocast():
                    logits, loss = model(x, y)
                loss = loss / grad_accum_steps
                loss_accum += accelerator.reduce(loss.detach(), reduction='mean')
                accelerator.backward(loss)
                gradient_accumulation_step += 1
                if gradient_accumulation_step == grad_accum_steps:
                    norm = model.get_global_grad_norm()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    t1 = time.time()
                    dt = (t1 - t0) * 1000
                    tokens_processed = B * T * grad_accum_steps * ddp_world_size
                    tokens_per_sec = (tokens_processed) / (t1 - t0)                    
                    progress_bar.update(1)
                    if accelerator.is_main_process:
                        tqdm.write(f"step {global_step:5d} | loss: {loss_accum.item():.6f} | lr {scheduler.get_last_lr()[0]:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f} | left time: {dt/1000*(total_steps-global_step-1):.2f}s")
                        with open(log_file, "a") as f:
                            f.write(f"{global_step} train {loss_accum.item():.6f}\n")
                    t0 = time.time()
                    loss_accum = 0.0
                    gradient_accumulation_step = 0
                    global_step += 1
                    if global_step % 250 == 0 or (global_step == total_steps - 1):
                        evaluate()
                        model.train()
                    if global_step >= total_steps:
                        break
    else:
        # generate some examples
        load_checkpoint_in_model(model, log_dir, device_map={"": device})
        model.eval()
        num_return_sequence = 4
        max_length = 32

        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        token = enc.encode("Hello, I'm a language model,")
        token = torch.tensor(token, dtype=torch.long)
        token = token.unsqueeze(0).repeat(num_return_sequence, 1)
        xgen = token.to(accelerator.device)
        # do not influence the seed of training
        sample_rng = torch.Generator(device=accelerator.device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.shape[1] < max_length:
            with torch.no_grad():
                logits, loss = model(xgen) # (B, T, vocab_size)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                # (5, 50257) -> (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # multinomial: random sample from a random distribution
                ix = torch.multinomial(topk_probs, num_samples=1, generator=sample_rng)
                # ix = 0, 1, ..., 49. get real index from topk_indices
                xcol = torch.gather(topk_indices, -1, ix)
                xgen = torch.cat((xgen, xcol), dim=-1)

        for i in range(num_return_sequence):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # free GPU/nccl
    if ddp:
        destroy_process_group()

if __name__ == '__main__':
    import argparse
    from types import SimpleNamespace
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_from_checkpoint", action="store_true")
    parser.add_argument("--use_compile", action="store_true")
    parser.add_argument("--auxiliary_loss", action="store_true")
    parser.add_argument("--model_type", type=str, default="GPT")
    parser.add_argument("--config_path", type=str, default="config")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "infer"])
    args = parser.parse_args()
    model_dict = {
        'GPT': GPT,
        'Qwen': Qwen,
        'DeepSeek': DeepSeek
    }
    config_path = f"model/{args.model_type}/{args.config_path}.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    # config[key] -> config.key
    config = SimpleNamespace(**config)
    config.config_path = args.config_path
    config.model_type = args.model_type
    config.use_compile = args.use_compile
    config.auxiliary_loss = args.auxiliary_loss
    config.mode = args.mode
    config.resume_from_checkpoint = args.resume_from_checkpoint
    main(model_dict[config.model_type], config)
