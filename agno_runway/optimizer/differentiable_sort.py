from __future__ import annotations

import torch


def sinkhorn(log_alpha: torch.Tensor, n_iters: int = 30) -> torch.Tensor:
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return torch.exp(log_alpha)


def soft_rank(scores: torch.Tensor, tau: float = 0.1, n_iters: int = 30) -> torch.Tensor:
    n = scores.size(-1)
    scores = scores.unsqueeze(-1)
    arange = torch.arange(n, device=scores.device).float().unsqueeze(-1)
    # NeuralSort-style pairwise differences.
    diff = scores - scores.transpose(-1, -2)
    log_alpha = -torch.abs(diff) / tau
    log_alpha = log_alpha - log_alpha.max(dim=-1, keepdim=True).values
    perm = sinkhorn(log_alpha, n_iters=n_iters)
    positions = torch.matmul(perm, arange).squeeze(-1)
    return positions
