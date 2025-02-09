import torch
import inspect
import torch.nn as nn
from torch.nn import functional as F


def precompute_freqs_cis(config):
    # 1/(base)^(2t/d), for embedding dimension, (dim//2,)
    dim = config.n_embd // config.n_head
    freqs = 1.0 / (config.base**(torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    # (block_size,)
    t = torch.arange(config.block_size, device=freqs.device)
    # (block_size, dim//2)
    freqs = torch.outer(t, freqs)
    # torch.polar: turn (r, theta) to r(cos(theta) + isin(theta))
    # (block_size, dim//2) -> (block_size, dim//2) (complex form)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def Qwen_apply_rope(x, freqs_cis):
    dtype = x.dtype
    B, T, n_head, head_dim = x.shape
    # (B, T, n_head, head_dim) -> (B, T, n_head, head_dim//2, 2) -> (B, T, n_head, head_dim//2) (complex form)
    # to align with the rope form of Qwen, who treats (x[0], x[d//2]) as (x, y) which the vanilla rope treat (x[0], x[1]) as (x, y)
    x = torch.view_as_complex(x.float().view(B, T, n_head, 2, -1).transpose(3, 4).contiguous())
    # (T, head_dim//2) (complex) -> (1, T, 1, head_dim//2) (complex form)
    freqs_cis = freqs_cis.view(1, T, 1, x.shape[-1])
    # (B, T, n_head, head_dim//2) -> (B, T, n_head, head_dim//2, 2) -> (B, T, n_head, head_dim)
    y = torch.view_as_real(x * freqs_cis).transpose(3, 4).flatten(3)
    return y.contiguous().to(dtype)

def repeated_kv(x, kv_group):
    B, T, n_head, head_dim = x.shape
    x = x.unsqueeze(-2).expand(B, T, n_head, kv_group, head_dim)
    x = x.reshape(B, T, n_head * kv_group, head_dim)
    return x

class RMSNorm(nn.Module):
    
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        #rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        #return x * rms * self.weight
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)

class QwenGroupedQueryAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.head_dim = config.n_embd // config.n_head
        self.n_q_head = config.n_head
        self.kv_group = config.kv_group
        self.n_kv_head = config.n_head // config.kv_group
        self.q_dim = config.n_embd
        self.kv_dim = self.n_kv_head * self.head_dim

        self.q_proj = nn.Linear(config.n_embd, self.q_dim)
        self.k_proj = nn.Linear(config.n_embd, self.kv_dim)
        self.v_proj = nn.Linear(config.n_embd, self.kv_dim)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        #self.c_attn = nn.Linear(config.n_embd, self.q_dim + 2 * self.kv_dim)
        #self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
    
    def forward(self, x, freqs_cis):
        B, T, hidden = x.shape
        #q, k, v = self.c_attn(x).split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = q.view(B, T, self.n_q_head, self.head_dim)
        k = k.view(B, T, self.n_kv_head, self.head_dim)
        v = v.view(B, T, self.n_kv_head, self.head_dim)

        # apply rope
        q = Qwen_apply_rope(q, freqs_cis).transpose(1, 2)
        k = Qwen_apply_rope(k, freqs_cis)
        k = repeated_kv(k, self.kv_group).transpose(1, 2)
        v = repeated_kv(v, self.kv_group).transpose(1, 2)

        # FlashAttention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().reshape(B, T, hidden)
        y = self.o_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.n_inter, bias=False)
        self.down_proj = nn.Linear(config.n_inter, config.n_embd, bias=False)
        self.up_proj = nn.Linear(config.n_embd, config.n_inter, bias=False)
    
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.n_embd)
        self.self_attn = QwenGroupedQueryAttention(config)
        self.post_attention_layernorm = RMSNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x, freqs_cis):
        x = x + self.self_attn(self.input_layernorm(x), freqs_cis)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

class Qwen(nn.Module):
    
    def __init__(self, config, process_rank):
        super().__init__()
        self.config = config
        self.model = nn.ModuleDict(dict(
            embed_tokens = nn.Embedding(config.vocab_size, config.n_embd, config.pad_token_id),
            layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            norm = RMSNorm(config.n_embd),
        ))
        # decode head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight sharing following gpt-2, Qwen did not implement this
        if config.weight_sharing:
            self.model.embed_tokens.weight = self.lm_head.weight
        self.register_buffer("freqs_cis", precompute_freqs_cis(config), persistent=False)
        self.master_process = process_rank == 0
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # a rough xavier: 0.2 \in (sqrt(1/1600), sqrt(1/768))
        if isinstance(module, nn.Linear):
            std = 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, f"Cannot forward sequence of length{T}, block size is only {self.config.block_size}"

        # tokenize
        # (B, T) -> (B, T, n_embd)
        x = self.model.embed_tokens(idx)
        # encode
        for block in self.model.layers:
            x = block(x, self.freqs_cis[:T])
        # norm
        x = self.model.norm(x)
        # decode
        # (B, T, n_embd) -> (B, T, vocab_size)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            # logits:  (B, T, vocab_size) -> (B*T, vocab_size)
            # targets: (B, T) -> (B*T,)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, config, model_type, process_rank):
        assert "Qwen" in model_type
        model = Qwen(config, process_rank)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith("freqs_cis")]

        from transformers import AutoModelForCausalLM
        model_hf = AutoModelForCausalLM.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_hf_keys = sd_hf.keys()

        assert len(sd_hf_keys) == len(sd_keys), f"mismatched keys: {len(sd_hf_keys)} != {len(sd_keys)}"
        for k in sd_hf_keys:
            assert sd[k].shape == sd_hf[k].shape
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
