"""Зведена таблиця з CSV результатів (по одному CSV на запуск).

    python -m cataclysm.eval.table results/*.csv
Групує за (policy, level, train_seed). Для навчених політик додає рядок "mean over seeds".
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict

import numpy as np

from cataclysm.eval.metrics import summarize


def _load(paths: list[str]) -> dict:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                row = {k: (float(v) if k not in ("policy", "level", "train_seed") and v != "" else v)
                       for k, v in r.items()}
                groups[(r["policy"], r["level"], r["train_seed"])].append(row)
    return groups


def main(paths: list[str]) -> None:
    groups = _load(paths)
    print("| policy | level | train_seed | episodes | success [95% CI] | steps(all) | coll_obs | coll_agent |")
    print("|---|---|---|---|---|---|---|---|")
    per_policy: dict[tuple, list[dict]] = defaultdict(list)
    for (pol, lvl, ts), rows in sorted(groups.items()):
        s = summarize(rows)
        print(f"| {pol} | {lvl} | {ts or '-'} | {s['episodes']} | {s['success_rate']:.3f} "
              f"[{s['success_ci_low']:.3f}, {s['success_ci_high']:.3f}] | {s['steps_mean_all']:.0f} | "
              f"{s['collisions_obstacle_mean']:.1f} | {s['collisions_agent_mean']:.1f} |")
        if ts != "":
            per_policy[(pol.split(":")[0], lvl)].append(s)
    for (pol, lvl), ss in sorted(per_policy.items()):
        if len(ss) > 1:
            sr = np.array([s["success_rate"] for s in ss])
            print(f"| {pol} (mean over {len(ss)} seeds) | {lvl} | - | {sum(s['episodes'] for s in ss)} | "
                  f"{sr.mean():.3f} +- {sr.std(ddof=1):.3f} | {np.mean([s['steps_mean_all'] for s in ss]):.0f} | "
                  f"{np.mean([s['collisions_obstacle_mean'] for s in ss]):.1f} | "
                  f"{np.mean([s['collisions_agent_mean'] for s in ss]):.1f} |")


if __name__ == "__main__":
    main(sys.argv[1:])
