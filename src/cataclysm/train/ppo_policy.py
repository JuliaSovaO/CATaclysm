"""Навчена політика як об'єкт з контрактом `Policy` (SPEC, розділ 4).

Використання: `python -m cataclysm.eval.run_eval --policy ppo:models/ippo_seed0/best.pt --level L1`
(`ppo-det:` означає вибір найімовірнішої дії замість семплування).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from cataclysm.policy import OBS_DIM
from cataclysm.train.common import ActorCritic, StackBuffer, load_checkpoint


class PPOPolicy:
    """Спільна політика для всіх роботів. Кожен робот отримує лише власне спостереження."""

    def __init__(self, net: ActorCritic, frame_stack: int, deterministic: bool = False, seed: int = 0):
        self.net = net
        self.k = int(frame_stack)
        self.deterministic = deterministic
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._stack: StackBuffer | None = None
        self._names: list[str] = []

    def reset(self) -> None:
        # Генератор не перезапускається: послідовність епізодів у межах одного запуску детермінована.
        self._stack = None
        self._names = []

    def act(self, obs: dict[str, np.ndarray]) -> dict[str, int]:
        names = sorted(obs)
        arr = np.stack([np.asarray(obs[n], dtype=np.float32) for n in names])
        if self._stack is None or names != self._names:
            self._stack = StackBuffer(len(names), self.k, OBS_DIM)
            self._stack.reset_rows(slice(None), arr)
            self._names = names
        else:
            self._stack.push(arr)
        with torch.no_grad():
            logits = self.net.actor(torch.from_numpy(self._stack.flat()))
        if self.deterministic:
            acts = logits.argmax(dim=-1).numpy()
        else:
            p = torch.softmax(logits.double(), dim=-1).numpy()
            u = self._rng.random(len(names))[:, None]
            acts = (p.cumsum(axis=1) < u).sum(axis=1).clip(max=p.shape[1] - 1)
        return {n: int(a) for n, a in zip(names, acts)}


def resolve_checkpoint(path: str | Path) -> Path:
    p = Path(path)
    if p.is_dir():
        for name in ("best.pt", "last.pt"):
            if (p / name).exists():
                return p / name
        raise FileNotFoundError(f"У {p} немає best.pt або last.pt")
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def load_policy(path: str | Path, cfg: dict | None = None, deterministic: bool = False, seed: int = 0) -> PPOPolicy:
    """`cfg` не потрібен (залишений для сумісності з run_eval)."""
    net, k, _ = load_checkpoint(resolve_checkpoint(path))
    return PPOPolicy(net, k, deterministic=deterministic, seed=seed)
