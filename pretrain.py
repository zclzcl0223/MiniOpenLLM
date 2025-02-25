import math
import torch
import inspect
import tiktoken
import time
import os
import json
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
from dataloader import DataLoader
from tqdm import trange

def unwrap_model(model):
    # Unwrap DDP
    if hasattr(model, 'module'):
        model = model.module
    # Unwrap torch.compile
    if hasattr(model, '_orig_mod'):
        model = model._orig_mod
    return model

def get_lr(it, warmup_steps, max_steps, max_lr, min_lr):
    # warm up
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # after cosine decay
    if it > max_steps:
        return min_lr
    # cosine decay
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

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
        assert torch.cuda.is_available(), "we need CUDA for DDP"
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

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # gradient accumulation
    total_batch_size = config.token_per_epoch # 2**19, ~0.5M, the number of tokens per batch
    B = config.batch_size
    T = config.block_size
    assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisble by B * T"
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
    torch.set_float32_matmul_precision('high')

    # log
    log_dir = f"log/{config.model_type}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")

    # dataset
    train_loader = DataLoader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
    val_loader = DataLoader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')

    # train from scratch or checkpoint
    resume_from_checkpoint = config.resume_from_checkpoint
    if resume_from_checkpoint:
        checkpoint_files = [f for f in os.listdir(log_dir) if f.startswith(f"model_{config.config_path.split('_')[1]}") 
                            and f.endswith(".pt")]
        assert len(checkpoint_files) > 0, "no checkpoints found"
        latest_checkpoint = sorted(checkpoint_files)[-1] # load the latest checkpoint
        checkpoint_path = os.path.join(log_dir, latest_checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # load model
        model = model(checkpoint['config'], process_rank=ddp_rank)
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        log_file = os.path.join(log_dir, f"log_{total_params:.0f}M.txt")
        model.to(device)
        model.load_state_dict(checkpoint['model'])
        # load optimizer
        optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=config.max_lr, device_type=device_type)
        optimizer.load_state_dict(checkpoint['optimizer'])
        # load train set state
        train_loader.set(checkpoint['train_loader'])
        # load step
        current_step = checkpoint['step'] + 1
        if master_process:
            print(f"resuming training from step {current_step} with a validation loss of {checkpoint['val_loss']:.4f}")
    else:
        # real vocab_size: 50257
        model = model(config, process_rank=ddp_rank)
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        model.to(device)
        current_step = 0
        # with weight decay
        optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=config.max_lr, device_type=device_type)
        log_file = os.path.join(log_dir, f"log_{total_params:.0f}M.txt")
        if config.mode == "train":
            with open(log_file, "w") as f: # clear log
                pass
    if master_process:
        print(f"Parameters: {total_params:.0f}M")
    use_compile = config.use_compile
    if use_compile:
        if master_process:
            print(f"compiling model...")
        model = torch.compile(model)
    if ddp:
        model = DDP(model, 
                    device_ids=[ddp_local_rank],
                    find_unused_parameters=True) # enable unused parameters for MoE training
    raw_model = unwrap_model(model)

    max_lr = config.max_lr
    min_lr = max_lr * 0.1
    warmup_steps = config.warmup_steps  # 375M / 2**19
    max_steps = config.max_steps  # 10B / 2**19


    if config.mode == "train":
        for step in range(current_step, max_steps):
            t0 = time.time()

            # eval on valiation set
            if step % 250 == 0 or (step == max_steps - 1):
                model.eval()
                val_loader.reset()
                with torch.no_grad():
                    val_loss_accum = 0.0
                    val_loss_steps = 20
                    for _ in range(val_loss_steps):
                        x, y = val_loader.next_batch()
                        x, y = x.to(device), y.to(device)
                        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                            logits, loss = model(x, y)
                        loss = loss / val_loss_steps
                        val_loss_accum += loss.detach()
                if ddp:
                    dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
                if master_process:
                    print(f"validation loss: {val_loss_accum.item():.4f}")
                    with open(log_file, "a") as f:
                        f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                    if step > 0 and (step % 2500 == 0 or step == max_steps - 1):
                        # save checkpoint
                        train_loader_checkpoint = {'current_shard': train_loader.current_shard, 'current_position': train_loader.current_position}
                        checkpoint_path = os.path.join(log_dir, f"model_{total_params:.0f}M_{step:05d}.pt")
                        checkpoint = {
                            'model': raw_model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'config': raw_model.config,
                            'step': step,
                            'val_loss': val_loss_accum.item(),
                            'train_loader': train_loader_checkpoint,
                
                        }
                        torch.save(checkpoint, checkpoint_path)

            # eval on hellaswag
            if step % 250 == 0 or (step == max_steps - 1):
                num_correct_norm = 0
                num_total = 0
                model_for_eval = raw_model
                model_for_eval.eval()
                for i, example in enumerate(iterate_examples("val")):
                    if i % ddp_world_size != ddp_rank:
                        continue
                    _, tokens, mask, label = render_example(example)
                    tokens = tokens.to(device)
                    mask = mask.to(device
                    )
                    with torch.no_grad():
                        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                            logits, loss = model_for_eval(tokens)
                        pred_norm = get_most_likely_row(tokens, mask, logits)
                    num_total += 1
                    num_correct_norm += int(pred_norm==label)
                if ddp:
                    num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                    num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
                    dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                    dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                    num_total = num_total.item()
                    num_correct_norm = num_correct_norm.item()
                acc_norm = num_correct_norm / num_total
                if master_process:
                    print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
                    with open(log_file, "a") as f:
                        f.write(f"{step} hella {acc_norm:.4f}\n")

            # train
            if step < max_steps - 1:
                model.train()
                optimizer.zero_grad()
                loss_accum = 0.0
                # gradient accumulation
                for micro_step in range(grad_accum_steps):
                    x, y = train_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    # only syn in last batch
                    if ddp:
                        model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    # smart breakpoints!!!
                    # import code; code.interact(local=locals())
                    loss = loss / grad_accum_steps
                    loss_accum += loss.detach()
                    # backward just adds gradient
                    loss.backward()
                if ddp:
                    dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
                # when use ddp, we clip the grad after syn
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                optimizer.step()
                # wait for gpu or t1 is the time that cpu sends tasks to gpu queue.
                torch.cuda.synchronize()
                t1 = time.time()
                dt = (t1 - t0) * 1000
                tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
                tokens_per_sec = (tokens_processed) / (t1 - t0)
                if master_process:
                    print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}",
                          f" | left time: {dt/1000*(max_steps-step-1):.2f}s")
                    with open(log_file, "a") as f:
                        f.write(f"{step} train {loss_accum.item():.6f}\n")
    else:
        # generate some examples
        model_for_eval = unwrap_model(model)
        model_for_eval.eval()
        num_return_sequence = 4
        max_length = 32

        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        token = enc.encode("Hello, I'm a language model,")
        token = torch.tensor(token, dtype=torch.long)
        token = token.unsqueeze(0).repeat(num_return_sequence, 1)
        xgen = token.to(device)
        # do not influence the seed of training
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.shape[1] < max_length:
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
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
