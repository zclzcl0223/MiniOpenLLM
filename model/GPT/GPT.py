from dataclasses import dataclass
import math
import torch
import inspect
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # q, k, v
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # self.register_buffer(
        #     "bias",
        #     torch.tril(
        #         torch.ones(config.block_size, config.block_size).reshape(1, 1, config.block_size, config.block_size)
        #     )
        # )
        #self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, hidden = x.shape
        q, k, v = torch.chunk(self.c_attn(x), 3, -1)
        # (B, T, hidden) -> (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        q = q.reshape(B, T, -1, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T, -1, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, -1, self.head_dim).transpose(1, 2)
        
        """
        # (B, n_head, T, head_dim) x (B, n_head, head_dim, T) -> (B, n_head, T, T)
        att_score = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att_score = att_score.masked_fill(self.bias[:, :, :T, :T]==0, float('-inf'))
        weight = F.softmax(att_score, dim=-1)
        # (B, n_head, T, T) x (B, n_head, T, head_dim) -> (B, n_head, T, head_dim)
        y = weight @ v
        """
        # FlashAttention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().reshape(B, T, hidden)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # no 'tanh' approximation anymore now
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        # pre-norm
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768 

class GPT(nn.Module):

    def __init__(self, config, process_rank):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        self.master_process = process_rank == 0

        # init params
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        # a rough xavier: 0.2 \in (sqrt(1/1600), sqrt(1/768))
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # deviation accumulation caused by residue: x + x + ... + x -> n*var(x)
                # fix it by: (1/sqrt(n)) (x + x + ... + x) -> var(x), where n is the number of residual connection
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, f"Cannot forward sequence of length{T}, block size is only {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        # (T) -> (T, n_embd)
        pos_emb = self.transformer.wpe(pos)
        # (B, T) -> (B, T, n_embd)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        # (B, T, n_embd) -> (B, T, vocab_size)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            # logits:  (B, T, vocab_size) -> (B*T, vocab_size)
            # targets: (B, T) -> (B*T,)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


    @classmethod
    def from_pretrained(config, model_type, process_rank):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel

        print("loading weighths from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 124M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 124M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 124M params
        }[model_type]
        config.n_layer = config_args[model_type]["n_layer"]
        config.n_layer = config_args[model_type]["n_head"]
        config.n_layer = config_args[model_type]["n_embd"]
        #config_args['vocab_size'] = 50257
        #config_args['block_size'] = 1024
        #config = GPTConfig(**config_args)
        model = GPT(config, process_rank)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        #model_hf = GPT2LMHeadModel(config=GPT2LMHeadModel.config_class())
        model_hf = GPT2LMHeadModel.from_pretrained('gpt2')
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            # the original checkpoint is in tensorflow, so the weight matrix is transposed
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # only 2D parameters will be decayed
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if self.master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device_type
        if self.master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
