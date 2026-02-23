from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.optim as optim

from agno_runway.data.separation_builder import build_separation_matrix
from agno_runway.optimizer.graph_model import AGNOModel, build_features
from agno_runway.optimizer.robust_refiner import refine_schedule_with_runways

# ──────────────────────────────────────────────────────────────────────────────
# Synthetic data generation
# ──────────────────────────────────────────────────────────────────────────────

WAKE_CLASSES = ["H", "M", "L"]
WAKE_MAP = {"H": 0, "M": 1, "L": 2}


def _generate_episode(
    n_flights: int,
    eta_range: tuple[float, float] = (0.0, 3600.0),
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Synthesise a random flight scenario.

    Returns:
        eta          – (n,) scheduled estimated arrival times in seconds
        velocity     – (n,) aircraft speed (m/s)
        altitude     – (n,) aircraft altitude (m)
        wake_onehot  – (n, 3) one-hot wake class encoding
        sep_matrix   – (n, n) minimum separation in seconds
    """
    wake_labels = [random.choice(WAKE_CLASSES) for _ in range(n_flights)]

    eta = torch.rand(n_flights, device=device) * (eta_range[1] - eta_range[0]) + eta_range[0]
    velocity = torch.rand(n_flights, device=device) * 100.0 + 60.0     # 60–160 m/s
    altitude = torch.rand(n_flights, device=device) * 1000.0            # 0–1000 m

    idx = torch.tensor([WAKE_MAP[w] for w in wake_labels], device=device)
    wake_onehot = torch.zeros(n_flights, 3, device=device)
    wake_onehot.scatter_(1, idx.unsqueeze(1), 1.0)

    sep_matrix = torch.tensor(
        build_separation_matrix(wake_labels), dtype=torch.float32, device=device
    )

    return eta, velocity, altitude, wake_onehot, sep_matrix


# ──────────────────────────────────────────────────────────────────────────────
# Ground-truth optimal rank via FCFS (minimise total delay greedily)
# ──────────────────────────────────────────────────────────────────────────────

def _optimal_order(
    eta: torch.Tensor,
    sep_matrix: torch.Tensor,
    runway_count: int = 2,
) -> torch.Tensor:
    """
    Return the flight indices sorted by ascending ETA (FCFS ground truth).
    FCFS is provably delay-optimal when all separation values are equal; here
    it gives a fast, always-feasible reference ranking.
    """
    return torch.argsort(eta)


# ──────────────────────────────────────────────────────────────────────────────
# Pairwise ranking loss (RankNet-style)
# ──────────────────────────────────────────────────────────────────────────────

def _pairwise_ranking_loss(
    scores: torch.Tensor,
    optimal_order: torch.Tensor,
) -> torch.Tensor:
    """
    For every ordered pair (i, j) where flight optimal_order[i] should come
    BEFORE flight optimal_order[j], penalise the model if it assigns a lower
    score to i than j.

    Loss per pair = log(1 + exp(score[j] - score[i]))   (RankNet cross-entropy)

    This is the standard learning-to-rank objective used in IR and scheduling.
    """
    n = optimal_order.numel()
    # rank[k] = position of flight k in the optimal schedule
    rank = torch.zeros(n, device=scores.device)
    for pos, flight_idx in enumerate(optimal_order):
        rank[flight_idx] = pos

    # Build pair differences efficiently with broadcasting
    # For pair (a, b): if rank[a] < rank[b], score[a] should be > score[b]
    s = scores.unsqueeze(1) - scores.unsqueeze(0)   # (n, n) score differences
    r = rank.unsqueeze(1) - rank.unsqueeze(0)        # (n, n) rank differences

    # Only consider pairs where a should rank before b (rank[a] < rank[b] → r < 0)
    mask = r < 0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=scores.device, requires_grad=True)

    # RankNet: log(1 + exp(score[b] - score[a])) = log(1 + exp(-s[a,b]))
    loss = torch.log1p(torch.exp(-s[mask]))
    return loss.mean()


# ──────────────────────────────────────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────────────────────────────────────

def _wake_bias_scores(
    scores: torch.Tensor,
    eta: torch.Tensor,
    wake_onehot: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Apply the same ETA + wake bias used in schedule_with_agnos for consistency."""
    eta_norm = (eta - eta.mean()) / (eta.std() + 1e-6)
    wake_bias = torch.tensor([-0.4, 0.0, 0.4], device=device)
    return scores - eta_norm + wake_onehot.matmul(wake_bias)


# ──────────────────────────────────────────────────────────────────────────────
# Main training entrypoint
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AGNO GNN on simulated runway scenarios.")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Number of synthetic flight scenarios to train on.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Alias for --episodes (backward compat).")
    parser.add_argument("--steps-per-episode", type=int, default=1,
                        help="Gradient steps per episode (default 1).")
    parser.add_argument("--min-flights", type=int, default=8,
                        help="Minimum flights per synthetic episode.")
    parser.add_argument("--max-flights", type=int, default=30,
                        help="Maximum flights per synthetic episode.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    episodes = args.epochs if args.epochs is not None else args.episodes

    root = Path(__file__).resolve().parent
    outputs = root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    model_path = outputs / "model.pt"

    device = _resolve_device(args.device)
    print(f"Using device: {device.type}")

    model = AGNOModel(feature_dim=6, hidden_dim=64).to(device)

    # Resume from checkpoint if available
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Resumed from checkpoint: {model_path}")
    else:
        print("Starting training from random initialisation.")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=episodes, eta_min=1e-5)

    best_loss = float("inf")
    running_loss = 0.0
    log_every = max(1, episodes // 20)   # print ~20 times

    print(f"\nTraining for {episodes} episodes "
          f"({args.min_flights}–{args.max_flights} flights each) ...\n")

    for ep in range(1, episodes + 1):
        n = random.randint(args.min_flights, args.max_flights)
        eta, velocity, altitude, wake_onehot, sep_matrix = _generate_episode(n, device=device)
        adj = (sep_matrix > 0).float()
        optimal_ord = _optimal_order(eta, sep_matrix)

        for _ in range(args.steps_per_episode):
            optimizer.zero_grad()
            features = build_features(eta, velocity, altitude, wake_onehot)
            scores = model(features, adj)
            scores = _wake_bias_scores(scores, eta, wake_onehot, device)
            loss = _pairwise_ranking_loss(scores, optimal_ord)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
        running_loss += loss.item()

        if ep % log_every == 0:
            avg = running_loss / log_every
            print(f"  Episode {ep:>4}/{episodes}  avg_loss={avg:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")
            if avg < best_loss:
                best_loss = avg
                torch.save(model.state_dict(), model_path)
                print(f"            ✓ Saved checkpoint (best loss so far)")
            running_loss = 0.0

    # Final save
    torch.save(model.state_dict(), model_path)
    print(f"\nTraining complete. Model saved to: {model_path}")
    print(f"Best avg loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
