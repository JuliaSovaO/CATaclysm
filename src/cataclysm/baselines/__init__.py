"""Baseline без навчання (SPEC, розділ 13). Усі бачать лише observation."""
from __future__ import annotations

from cataclysm.baselines.random_policy import RandomPolicy
from cataclysm.baselines.scan_go import ScanAndGo

BASELINES = ("random", "greedy", "reactive")


def make_baseline(name: str, cfg: dict, seed: int = 0):
    """Створює baseline, підганяючи кількість кроків оберту під швидкість повороту з cfg."""
    omega = float(cfg["agents"]["omega_deg_s"]) * float(cfg["world"]["dt_s"])
    scan_steps = max(4, int(round(360.0 / omega)))
    if name == "random":
        return RandomPolicy(seed=seed)
    if name == "greedy":  # baseline (1): без координації
        return ScanAndGo(scan_steps=scan_steps, separate=False, seed=seed)
    if name == "reactive":  # baseline (2): ручні правила, Boids-стиль
        return ScanAndGo(scan_steps=scan_steps, separate=True, seed=seed)
    raise ValueError(f"Невідомий baseline: {name}. Доступні: {BASELINES}")
