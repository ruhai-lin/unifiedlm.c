import inspect
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernel import CHUNK, unified_memory


@dataclass
class ModelArgs:
    dim: int = 288
    n_layers: int = 12          # number of UnifiedBlocks
    hidden_dim: int = 712       # memory slots per block
    vocab_size: int = 32000
    max_seq_len: int = 256
    norm_eps: float = 1e-5
    dropout: float = 0.0
    conv_size: int = 4          # causal depthwise conv on the address path
    prior: float = 4.0          # initial prior mass w of every slot, in tokens


def _hi(x):
    """Slot statistics stay in fp32 even under bf16/fp16 autocast."""
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(_hi(x)).type_as(x)
        return output * self.weight


class UnifiedBlock(nn.Module):
    """
    One associative memory whose slots hold running posterior means.

        address  r_t = normalize(relu(K u_t + b)^2),  u_t = causal depthwise conv(RMSNorm(x_t))
        write    key r_{t-1}, value v_t = W_v h_t
        read     y_t = sum_a r_t,a * (w_a mu_a + sum_{s<=t} lam^(t-s) r_{s-1,a} v_s)
                                   / (w_a      + sum_{s<=t} lam^(t-s) r_{s-1,a})

    w_a -> inf: slot a is a static FFN slot (mu = W_down). w_a -> 0: a context register.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        d, m = args.dim, args.hidden_dim
        self.norm = RMSNorm(d, eps=args.norm_eps)
        self.conv = nn.Conv1d(d, d, args.conv_size, groups=d, padding=args.conv_size - 1, bias=False)
        self.key = nn.Linear(d, m, bias=True)          # slot addresses and thresholds
        self.mu = nn.Linear(m, d, bias=False)          # slow slot contents
        self.log_eps = nn.Parameter(torch.zeros(()))    # implicit null slot
        self.val = nn.Linear(d, d, bias=False)          # value written into the fast store
        self.log_rate = nn.Parameter(torch.tensor(math.log(1.0 / (4 * args.max_seq_len))))  # forgetting
        self.log_w = nn.Parameter(torch.full((m,), math.log(args.prior)))                   # prior mass
        self.dropout = nn.Dropout(args.dropout)

    def code(self, u):
        z = F.relu(_hi(self.key(u))).square()
        return z / (self.log_eps.clamp(-10, 10).exp() + z.sum(-1, keepdim=True))

    def _w(self):
        return self.log_w.clamp(-7.0, 12.0).exp()

    def forward(self, x):
        h = self.norm(x)
        u = self.conv(h.transpose(1, 2))[..., : h.size(1)].transpose(1, 2)
        r = self.code(u)                                   # (B,T,m) read address
        v = self.val(h)
        # fast read (Triton): q = r / c and sum_{s<=t} lam^(t-s) (q_t . r_{s-1}) v_s, see kernel.py
        w = self._w()
        q, fast = unified_memory(r, v, -self.log_rate.exp(), w)
        return x + self.dropout(self.mu((q * w).type_as(x)) + fast.type_as(x))

    # ---- token-by-token form: the reference for run.c ----
    def init_state(self, batch: int, device):
        d, m = self.mu.out_features, self.mu.in_features
        dt = self.mu.weight.dtype
        return {"buf": torch.zeros(batch, d, self.conv.kernel_size[0] - 1, device=device, dtype=dt),
                "U": (self.mu.weight * self._w()).expand(batch, d, m).clone(),   # slot contents
                "c": self._w().expand(batch, m).clone(),                          # slot evidence mass
                "r_prev": torch.zeros(batch, m, device=device, dtype=dt)}

    @torch.no_grad()
    def step(self, x, s):
        """x: (B,d), one position. Write (key r_{t-1}, value v_t), then read."""
        h = self.norm(x)
        win = torch.cat([s["buf"], h[:, :, None]], dim=-1)  # oldest -> newest
        u = (win * self.conv.weight[:, 0, :]).sum(-1)
        s["buf"] = win[:, :, 1:]
        r = self.code(u)
        lam = torch.exp(-self.log_rate.exp())
        v = self.val(h)
        w = self._w()
        s["U"] = lam * s["U"] + (1 - lam) * self.mu.weight * w + v[:, :, None] * s["r_prev"][:, None, :]
        s["c"] = lam * s["c"] + (1 - lam) * w + s["r_prev"]
        y = (s["U"] @ (r / s["c"])[:, :, None]).squeeze(-1)
        s["r_prev"] = r
        return x + y.to(x.dtype)


class UnifiedLM(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = nn.ModuleList([UnifiedBlock(params) for _ in range(params.n_layers)])
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        self.tok_embeddings.weight = self.output.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("mu.weight"):       # residual projection, GPT-2 style scaling
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(params.n_layers))
            elif pn.endswith("val.weight"):    # start with an empty fast store
                nn.init.zeros_(p)
            elif pn.endswith("conv.weight"):   # start as identity (u_t = h_t)
                with torch.no_grad():
                    p.zero_()
                    p[:, 0, -1] = 1.0
        self.last_loss = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.dropout(self.tok_embeddings(tokens))
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        if targets is not None:
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.output(h[:, [-1], :])
            self.last_loss = None

        return logits

    @torch.no_grad()
    def forward_recurrent(self, tokens: torch.Tensor) -> torch.Tensor:
        """Token-by-token inference with constant state per block, exactly what run.c computes."""
        B, T = tokens.shape
        states = [layer.init_state(B, tokens.device) for layer in self.layers]
        out = []
        for t in range(T):
            h = self.tok_embeddings(tokens[:, t])
            for layer, s in zip(self.layers, states):
                h = layer.step(h, s)
            out.append(self.output(self.norm(h)))
        return torch.stack(out, dim=1)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        N = sum(p.numel() for p in self.parameters())
        cfg = self.params
        T, m, d = cfg.max_seq_len, cfg.hidden_dim, cfg.dim
        mix = min(CHUNK, T) * (2 * m + d) + 2 * m * d          # in-chunk mixing + carried state
        flops_per_token = 6 * N + 6 * cfg.n_layers * mix
        flops_per_iter = flops_per_token * T * fwdbwd_per_iter
        return flops_per_iter * (1.0 / dt) / 312e12

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            if temperature == 0.0:
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
