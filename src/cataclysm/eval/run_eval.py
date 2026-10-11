"""Оцінювання політики на фіксованих тестових seed (SPEC, розділ 13).

Приклад:
    python -m cataclysm.eval.run_eval --policy greedy --level L1 --episodes 200 --out results/L1_greedy.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from cataclysm.baselines import BASELINES, make_baseline
from cataclysm.env import SwarmEnv, load_env_config
from cataclysm.eval.metrics import CSV_COLUMNS, run_episode, summarize

TEST_SEED0 = 10000  # тестові seed 10000.., тренувальні 0..9999


def make_policy(name: str, cfg: dict, seed: int = 0):
    if name in BASELINES:
        return make_baseline(name, cfg, seed)
    if name.startswith("ppo:"):
        try:
            from cataclysm.train.ppo_policy import load_policy  # реалізує Роль B
        except ImportError as e:
            raise ImportError("Для 'ppo:<шлях>' потрібен cataclysm.train.ppo_policy.load_policy (Роль B)") from e
        return load_policy(name[4:], cfg)
    raise ValueError(f"Невідома політика: {name}")


def evaluate(policy, cfg: dict, policy_name: str, level: str, episodes: int = 200,
             seed0: int = TEST_SEED0, train_seed: int | None = None) -> list[dict]:
    env = SwarmEnv(cfg)
    rows = []
    for k in range(episodes):
        row = run_episode(env, policy, seed0 + k)
        row.update({"policy": policy_name, "level": level, "train_seed": "" if train_seed is None else train_seed})
        rows.append(row)
    return rows


def write_csv(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in CSV_COLUMNS})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", required=True, help="random | greedy | reactive | ppo:<шлях>")
    ap.add_argument("--level", default="L0")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=TEST_SEED0)
    ap.add_argument("--train-seed", type=int, default=None)
    ap.add_argument("--n-hunters", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    overrides = {"agents": {"n_hunters": args.n_hunters}} if args.n_hunters else None
    cfg = load_env_config(args.level, overrides)
    policy = make_policy(args.policy, cfg, seed=0)
    rows = evaluate(policy, cfg, args.policy, args.level, args.episodes, args.seed0, args.train_seed)
    if args.out:
        write_csv(rows, args.out)
    s = summarize(rows)
    print(f"{args.policy} @ {args.level}: success {s['success_rate']:.3f} "
          f"[{s['success_ci_low']:.3f}, {s['success_ci_high']:.3f}], "
          f"steps(all) {s['steps_mean_all']:.0f}, steps(success) {s['steps_mean_success']:.0f}, "
          f"coll_obs {s['collisions_obstacle_mean']:.2f}, coll_agent {s['collisions_agent_mean']:.2f}")


if __name__ == "__main__":
    main()
