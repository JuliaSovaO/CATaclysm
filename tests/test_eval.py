import csv

import numpy as np

from cataclysm.baselines import make_baseline
from cataclysm.env import SwarmEnv, load_env_config
from cataclysm.eval.metrics import CSV_COLUMNS, summarize, wilson_interval
from cataclysm.eval.run_eval import evaluate, write_csv
from cataclysm.policy import OBS_DIM
from cataclysm.train.dummy_env import DummyEnv


def test_wilson_interval_sane():
    lo, hi = wilson_interval(50, 100)
    assert 0.39 < lo < 0.5 < hi < 0.61
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_run_eval_writes_csv_with_expected_columns(tmp_path):
    cfg = load_env_config("L0")
    rows = evaluate(make_baseline("random", cfg), cfg, "random", "L0", episodes=3)
    out = tmp_path / "r.csv"
    write_csv(rows, out)
    with open(out, encoding="utf-8") as f:
        rd = list(csv.DictReader(f))
    assert list(rd[0].keys()) == CSV_COLUMNS and len(rd) == 3


def test_same_seed_same_metrics():
    cfg = load_env_config("L1")
    a = evaluate(make_baseline("greedy", cfg, 0), cfg, "greedy", "L1", episodes=5, seed0=10000)
    b = evaluate(make_baseline("greedy", cfg, 0), cfg, "greedy", "L1", episodes=5, seed0=10000)
    assert a == b


def test_scan_and_go_beats_random_on_L0():
    cfg = load_env_config("L0")
    g = summarize(evaluate(make_baseline("greedy", cfg), cfg, "greedy", "L0", episodes=20))
    r = summarize(evaluate(make_baseline("random", cfg), cfg, "random", "L0", episodes=20))
    assert g["success_rate"] >= 0.8 and g["success_rate"] > r["success_rate"]


def test_policy_sees_only_obs():
    cfg = load_env_config("L0")
    env = SwarmEnv(cfg)
    obs, _ = env.reset(seed=0)
    assert all(v.shape == (OBS_DIM,) for v in obs.values())
    acts = make_baseline("reactive", cfg).act(obs)
    assert set(acts) == set(obs) and all(a in (0, 1, 2, 3) for a in acts.values())


def test_dummy_env_matches_interface():
    env = DummyEnv(n_hunters=2, max_steps=3)
    obs, _ = env.reset(seed=0)
    assert obs["hunter_0"].shape == (OBS_DIM,)
    n = 0
    while env.agents:
        env.step({a: 0 for a in env.agents})
        n += 1
    assert n == 3
