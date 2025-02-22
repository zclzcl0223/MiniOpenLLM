import numpy as np
import tiktoken
import torch
import os
from torch.utils.data import IterableDataset, DataLoader

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class TokenDataset(IterableDataset):
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}
        self.rng = np.random.default_rng(1337)
        self.split = split

        # get the shard filename
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s] # load train or val
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        self.master_process = process_rank == 0
        if self.master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def load_shard(self, filename): # added from PR to avoid periodisation in training: https://github.com/karpathy/build-nanogpt/pull/52/files
        shard = load_tokens(filename)
        if self.split == 'train':  # shuffle document inside each shard
            # split tokens into documents using the <|endoftext|> special token and shuffle
            eot_positions = (torch.where(shard == enc.eot_token)[0] + 1).tolist()
            documents = [shard[start:end] for start, end in zip([0] + eot_positions[:-1], eot_positions)]
            self.rng.shuffle(documents)
            shard = torch.cat(documents) # concatenate the documents back together
        return shard

    def reset(self):
        self.current_shard = 0
        if self.split == 'train':  # shuffle shards
            self.rng.shuffle(self.shards)
        self.tokens = self.load_shard(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def set(self, loader_checkpoint):
        self.current_position = loader_checkpoint['current_position'] + self.B * self.T * self.process_rank
        self.current_shard = loader_checkpoint['current_shard']
        self.tokens = load_tokens(self.shards[self.current_shard])

    def sample_batches(self, tokens):
        total_tokens = len(tokens)
        idx = self.B * self.T * self.process_rank
        while idx + self.T + 1 < total_tokens:
            buf = tokens[idx: idx + self.T + 1]
            x = buf[:-1]
            y = buf[1:]
            yield x, y
            idx += self.T

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            shard_indices = range(len(self.shards))
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            shard_indices = range(worker_id, len(self.shards), num_workers)

        for i in shard_indices:
            tokens = self.load_shard(self.shards[i])
            yield from self.sample_batches(tokens)
        self.reset()
    
def get_dataloader(B, T, process_rank, num_processes, split, num_workers=0):
    dataset = TokenDataset(B, T, process_rank, num_processes, split)
    dataloader = DataLoader(dataset, batch_size=B, num_workers=0, pin_memory=False, shuffle=False)
    return dataloader
