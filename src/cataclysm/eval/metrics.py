"""Метрики й зведення результатів (SPEC, розділ 13)."""
from __future__ import annotations

import math

import numpy as np

CSV_COLUMNS = [
    "policy",
    "level",
    "train_seed",
    "episode_seed",
    "success",
    "steps",
    "collisions_obstacle",
    "collisions_agent",
    "n_hunters_at_capture",
]


def run_episode(env, policy, seed: int) -> dict:
    """Один епізод. Політика бачить лише observation."""
    obs, _ = env.reset(seed=seed)
    policy.reset()
    coll_obs = coll_agent = 0
    success, n_at_capture = False, 0
    while env.agents:
        actions = policy.act(obs)
        obs, _, _, _, infos = env.step(actions)
        for info in infos.values():
            coll_obs += int(info.get("collided_obstacle", False))
            coll_agent += int(info.get("collided_agent", False))
            if info.get("captured"):
                success = True
                n_at_capture = info["n_hunters_at_capture"]
    return {
        "episode_seed": seed,
        "success": int(success),
        "steps": env.t,
        "collisions_obstacle": coll_obs,
        "collisions_agent": coll_agent,
        "n_hunters_at_capture": n_at_capture,
    }


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Довірчий інтервал Вілсона для частки успіху."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    succ = np.array([r["success"] for r in rows], dtype=float)
    steps = np.array([r["steps"] for r in rows], dtype=float)
    lo, hi = wilson_interval(int(succ.sum()), n)
    return {
        "episodes": n,
        "success_rate": float(succ.mean()) if n else 0.0,
        "success_ci_low": lo,
        "success_ci_high": hi,
        "steps_mean_all": float(steps.mean()) if n else 0.0,  # невдалі = max_steps (цензура)
        "steps_mean_success": float(steps[succ > 0].mean()) if succ.sum() else float("nan"),
        "collisions_obstacle_mean": float(np.mean([r["collisions_obstacle"] for r in rows])) if n else 0.0,
        "collisions_agent_mean": float(np.mean([r["collisions_agent"] for r in rows])) if n else 0.0,
    }
