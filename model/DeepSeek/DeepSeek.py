import torch
import inspect
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F

def precompute_freqs_cis(config):
    # 1/(base)^(2t/d), for embedding dimension, (dim//2,)
    dim = config.qk_rope_head_dim
    freqs = 1.0 / (config.base**(torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    # (block_size,)
    t = torch.arange(config.block_size, device=freqs.device)
    # (block_size, dim//2)
    freqs = torch.outer(t, freqs)
    # torch.polar: turn (r, theta) to r(cos(theta) + isin(theta))
    # (block_size, dim//2) -> (block_size, dim//2) (complex form)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rope(x, freqs_cis):
    dtype = x.dtype
    B, T, n_head, head_dim = x.shape
    # (B, T, n_head, head_dim) -> (B, T, n_head, head_dim//2, 2) -> (B, T, n_head, head_dim//2) (complex form)
    x = torch.view_as_complex(x.float().view(B, T, n_head, -1, 2))
    # (T, head_dim//2) (complex) -> (1, T, 1, head_dim//2) (complex form)
    freqs_cis = freqs_cis.view(1, T, 1, x.shape[-1])
    # (B, T, n_head, head_dim//2) -> (B, T, n_head, head_dim//2, 2) -> (B, T, n_head, head_dim)
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)

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

class MultiHeadLatentAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        # kv, q
        self.n_head = config.n_head
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim

        self.latent_proj = nn.Linear(config.n_embd, self.q_lora_rank+self.kv_lora_rank+self.qk_rope_head_dim, bias=False)
        self.q_norm = RMSNorm(self.q_lora_rank)
        self.q_up_proj = nn.Linear(self.q_lora_rank, self.n_head * self.qk_head_dim, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.kv_up_proj = nn.Linear(self.kv_lora_rank, self.n_head * (self.qk_nope_head_dim + self.v_head_dim), bias=False)

        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x, freqs_cis):
        B, T, hidden = x.shape
        # (B, T, hidden) -> (B, T, q_lora_rank), (B, T, kv_lora_rank), (B, T, qk_rope_head_dim)
        q, kv, k_rope = torch.split(self.latent_proj(x), [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        # (B, T, q_lora_rank) -> (B, T, n_head * qk_nope_head_dim + n_head * qk_rope_head_dim) ->
        # (B, T, n_head, qk_nope_head_dim + qk_rope_head_dim)
        q = self.q_up_proj(self.q_norm(q)).view(B, T, self.n_head, -1)
        # (B, T, n_head * qk_nope_head_dim), (B, T, n_head * qk_rope_head_dim)
        q, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_rope = apply_rope(q_rope, freqs_cis)
        # (B, T, n_head * qk_nope_head_dim), (B, T, n_head * qk_rope_head_dim) -> 
        # (B, T, n_head, qk_nope_head_dim + qk_rope_head_dim) -> (B, n_head, T, qk_nope_head_dim + qk_rope_head_dim)
        q = torch.cat([q, q_rope], dim=-1).transpose(1, 2)
        
        # (B, T, kv_lora_rank) -> (B, T, n_head, qk_nope_head_dim + v_head_dim)
        kv = self.kv_up_proj(self.kv_norm(kv)).view(B, T, self.n_head, -1)
        # (B, T, n_head, qk_nope_head_dim + v_head_dim) -> (B, T, n_head, qk_nope_head_dim), (B, T, n_head, v_head_dim)
        k, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        # (B, T, 1, qk_rope_head_dim)
        k_rope = apply_rope(k_rope.unsqueeze(-2), freqs_cis).repeat(1, 1, self.n_head, 1)
        # (B, T, n_head, qk_nope_head_dim), (B, T, 1, qk_rope_head_dim) ->
        # (B, T, n_head, qk_nope_head_dim + qk_rope_head_dim) -> (B, n_head, T, qk_nope_head_dim + qk_rope_head_dim)
        k = torch.cat([k, k_rope], dim=-1).transpose(1, 2)
        # (B, T, hidden) -> (B, T, n_head, v_head_dim) -> (B, n_head, T, v_head_dim)
        v = v.reshape(B, T, self.n_head, -1).transpose(1, 2)

        # FlashAttention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # out_proj
        y = y.transpose(1, 2).contiguous().reshape(B, T, hidden)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config, inter_dim):
        super().__init__()
        self.w1 = nn.Linear(config.n_embd, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, config.n_embd, bias=False)
        self.w3 = nn.Linear(config.n_embd, inter_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x)), None

class MLP_Expert(nn.Module):

    def __init__(self, config, inter_dim):
        super().__init__()
        self.w1 = nn.Linear(config.n_embd, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, config.n_embd, bias=False)
        self.w3 = nn.Linear(config.n_embd, inter_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Gate(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.topk = config.n_activated_experts
        self.n_groups = config.n_expert_groups
        self.route_scale = config.route_scale
        self.score_func = config.score_func
        self.weight = nn.Parameter(torch.ones(self.n_embd, config.n_routed_experts))
        # bias for expert selection
        self.bias = nn.Parameter(torch.zeros(config.n_routed_experts)) if self.n_embd == 7168 else None
    
    def forward(self, x):
        scores = x @ self.weight
        if self.score_func == 'softmax':
            scores = F.softmax(scores, dim=-1)
        else:
            scores = F.sigmoid(scores)
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.shape[0], self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1) # get the largest value of dim -1
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = torch.zeros_like(scores[..., 0]).scatter_(1, indices, True)
            scores = (scores * mask.unsqueeze(-1)).flatten(1)
        # topk experts for each token, (B*T, n_activated_experts)
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)

        if self.score_func == 'sigmoid':
            weights /= (weights.sum(dim=-1, keepdim=True) + 1e-6)
        weights *= self.route_scale
        return original_scores, weights.type_as(x), indices

class DeepSeekMoE(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.moe_inter_dim = config.moe_inter_dim
        self.n_routed_experts = config.n_routed_experts
        #self.n_local_routed_experts = config.n_routed_experts // world_size
        #self.local_expert_start_idx = rank * self.n_local_routed_experts
        #self.local_expert_end_idx = self.local_expert_start_idx + self.n_local_routed_experts
        self.n_shared_experts = config.n_shared_experts
        self.n_activated_experts = config.n_activated_experts
        self.auxiliary_loss = config.auxiliary_loss
        # shared
        self.shared_experts = MLP(config, self.moe_inter_dim * self.n_shared_experts)
        # routed
        self.routed_experts = nn.ModuleList([MLP_Expert(config, self.moe_inter_dim) for _ in range(self.n_routed_experts)])
        #self.routed_experts = nn.ModuleList([MLP(config, self.moe_inter_dim) if self.local_expert_start_idx <= i < self.local_expert_end_idx 
        #                                     else None for i in range(self.n_routed_experts)])
        self.gated = Gate(config)

    # def forward_old(self, x):
    #     shape = x.shape
    #     x = x.view(-1, self.n_embd)
    #     y = self.shared_experts(x)
    #     # (B*T, n_activated_experts)
    #     weights, indices = self.gated(x)
    #     # (n_routed_experts,)
    #     usage_count = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
    #     for i, expert in enumerate(self.routed_experts):
    #         if usage_count[i] == 0:
    #             continue
    #         # (B*T, 1): the position of tokens and weights that use expert[i]
    #         idx, top = torch.where(indices == i)
    #         # (-, dim) * (-, 1) -> (-, dim)
    #         y[idx] += expert(x[idx]) * weights[idx, top, None]
    #     return y.view(shape)

    # with MoE auxiliary loss
    def forward(self, x):
        shape = x.shape
        B, T = shape[0], shape[1]
        x = x.view(-1, self.n_embd)
        y, _ = self.shared_experts(x)
        # (B*T, n_activated_experts)
        original_scores, weights, indices = self.gated(x)
        # (n_routed_experts,)
        usage_count = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        loss_bal = None
        for i, expert in enumerate(self.routed_experts):
            if usage_count[i] == 0:
                continue
            # (B*T, 1): the position of tokens and weights that use expert[i]
            idx, top = torch.where(indices == i)
            # (-, dim) * (-, 1) -> (-, dim)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        if self.training and self.auxiliary_loss:
            # (B*T, N)
            mask = torch.zeros_like(original_scores, dtype=torch.bool).scatter_(1, indices, True)
            f = (self.n_routed_experts / self.n_activated_experts) * torch.mean(original_scores * mask, dim=0) # (N,) 

            s = original_scores / torch.sum(original_scores, dim=-1, keepdim=True) # (B*T, N)
            p = torch.mean(s, dim=0) # (N,)
            loss_bal = torch.sum(p * f, dim=-1) # (1,)
        return y.view(shape), loss_bal
    
    # ToDo: allocate experts to different GPUs
    # def forward(self, x):
    #     shape = x.shape
    #     x = x.view(-1, self.n_embd)
    #     y = torch.zeros_like(x)
    #     # (B*T, n_activated_experts)
    #     weights, indices = self.gated(x)
    #     # (n_routed_experts,)
    #     usage_count = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
    #     for i in range(self.local_expert_start_idx, self.local_expert_end_idx):
    #         if usage_count[i] == 0:
    #             continue
    #         # (B*T, 1): the position of tokens and weights that use expert[i]
    #         idx, top = torch.where(indices == i)
    #         # (-, dim) * (-, 1) -> (-, dim)
    #         y[idx] += self.routed_experts[i](x[idx]) * weights[idx, top, None]
    #     if world_size > 1:
    #         dist.all_reduce(y, op=dist.ReduceOp.SUM)
    #     return (y + self.shared_experts(x)).view(shape)

class Block(nn.Module):

    def __init__(self, layer_id, config):
        super().__init__()
        self.rmsn1 = RMSNorm(config.n_embd)
        self.mla = MultiHeadLatentAttention(config)
        self.rmsn2 = RMSNorm(config.n_embd)
        self.ffn = MLP(config, config.n_embd*4) if layer_id < config.n_dense_layers else DeepSeekMoE(config)
    
    def forward(self, x, freqs_cis):
        x = x + self.mla(self.rmsn1(x), freqs_cis)
        out, loss_bal = self.ffn(self.rmsn2(x))
        x = x + out
        return x, loss_bal

class DeepSeek(nn.Module):

    def __init__(self, config, process_rank):
        super().__init__()
        self.config = config
        global world_size, rank
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.auxiliary_loss = config.auxiliary_loss
        self.alpha = config.alpha
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(i, config) for i in range(config.n_layer)])
        ))
        self.rmsn = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.register_buffer("freqs_cis", precompute_freqs_cis(config), persistent=False)
        # weight sharing following gpt-2, but deepseek did not implement weight sharing
        self.transformer.wte.weight = self.lm_head.weight
        self.master_process = process_rank == 0
        # init
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

    # def forward_old(self, idx, targets=None):
    #     B, T = idx.shape
    #     assert T <= self.config.block_size, f"Cannot forward sequence of length{T}, block size is only {self.config.block_size}"

    #     # (B, T) -> (B, T, n_embd)
    #     x = self.transformer.wte(idx)
    #     for block in self.transformer.h:
    #         x = block(x, self.freqs_cis[:T])
    #     x = self.rmsn(x)
    #     # (B, T, n_embd) -> (B, T, vocab_size)
    #     logits = self.lm_head(x)
    #     loss = None
    #     if targets is not None:
    #         # logits:  (B, T, vocab_size) -> (B*T, vocab_size)
    #         # targets: (B, T) -> (B*T,) -> (1,)
    #         loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    #     return logits, loss

    # with MoE auxiliary loss
    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, f"Cannot forward sequence of length{T}, block size is only {self.config.block_size}"

        # (B, T) -> (B, T, n_embd)
        x = self.transformer.wte(idx)
        loss_bal = 0
        for block in self.transformer.h:
            x, loss_bal_block = block(x, self.freqs_cis[:T])
            if loss_bal_block is not None:
                loss_bal += loss_bal_block
        x = self.rmsn(x)
        # (B, T, n_embd) -> (B, T, vocab_size)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            # logits:  (B, T, vocab_size) -> (B*T, vocab_size)
            # targets: (B, T) -> (B*T,)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) + self.alpha * loss_bal
        return logits, loss

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
