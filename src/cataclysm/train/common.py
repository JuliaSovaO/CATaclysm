"""Спільні частини навчання й інференсу: стек кадрів, мережа, збереження чекпоінтів.

Один і той самий код використовується під час навчання і в `ppo_policy.py`,
тому політика на тесті бачить вхід точно такого ж вигляду, як під час навчання.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from cataclysm.policy import N_ACTIONS, OBS_DIM


class StackBuffer:
    """Остання k спостережень для n потоків (агентів). Порядок: від старішого до новішого.

    На початку епізоду старіші кадри заповнені нулями.
    """

    def __init__(self, n: int, k: int, d: int = OBS_DIM):
        self.n, self.k, self.d = n, k, d
        self.buf = np.zeros((n, k, d), dtype=np.float32)

    def reset_rows(self, rows, obs: np.ndarray) -> None:
        self.buf[rows] = 0.0
        self.buf[rows, -1] = obs

    def push(self, obs: np.ndarray) -> None:
        self.buf = np.roll(self.buf, -1, axis=1)
        self.buf[:, -1] = obs

    def flat(self) -> np.ndarray:
        return self.buf.reshape(self.n, self.k * self.d).copy()


def _act(name: str) -> nn.Module:
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Невідома активація: {name}")


def _mlp(inp: int, hidden: list[int], out: int, activation: str, last_gain: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = inp
    for h in hidden:
        lin = nn.Linear(prev, h)
        nn.init.orthogonal_(lin.weight, np.sqrt(2))
        nn.init.zeros_(lin.bias)
        layers += [lin, _act(activation)]
        prev = h
    last = nn.Linear(prev, out)
    nn.init.orthogonal_(last.weight, last_gain)
    nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Окремі мережі актора й критика. Критик бачить лише локальний вхід (IPPO)."""

    def __init__(self, in_dim: int, n_actions: int = N_ACTIONS, hidden: list[int] | None = None,
                 activation: str = "tanh"):
        super().__init__()
        hidden = list(hidden or [64, 64])
        self.spec = {"in_dim": int(in_dim), "n_actions": int(n_actions), "hidden": [int(h) for h in hidden],
                     "activation": activation}
        self.actor = _mlp(in_dim, hidden, n_actions, activation, last_gain=0.01)
        self.critic = _mlp(in_dim, hidden, 1, activation, last_gain=1.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(x), self.critic(x).squeeze(-1)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x).squeeze(-1)


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
                last_value: torch.Tensor, gamma: float, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE(lambda). Форма [T, N]. dones[t] = 1, якщо епізод завершився після кроку t.

    Для обрізання за часом бонус gamma*V(кінцевого стану) уже доданий до rewards.
    """
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(last_value)
    for t in reversed(range(T)):
        next_v = last_value if t == T - 1 else values[t + 1]
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * nonterm - values[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + values


def save_checkpoint(path: str | Path, net: ActorCritic, frame_stack: int, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(), "spec": net.spec, "frame_stack": int(frame_stack),
                "obs_dim": OBS_DIM, "meta": meta or {}}, path)


def load_checkpoint(path: str | Path) -> tuple[ActorCritic, int, dict]:
    ck = torch.load(Path(path), map_location="cpu", weights_only=True)
    if ck["obs_dim"] != OBS_DIM:
        raise ValueError(f"Чекпоінт навчений для obs_dim={ck['obs_dim']}, а зараз OBS_DIM={OBS_DIM}")
    net = ActorCritic(**ck["spec"])
    net.load_state_dict(ck["model"])
    net.eval()
    return net, int(ck["frame_stack"]), ck.get("meta", {})
