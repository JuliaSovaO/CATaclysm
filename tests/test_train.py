import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cataclysm.env import SwarmEnv, load_env_config  # noqa: E402
from cataclysm.eval.metrics import run_episode  # noqa: E402
from cataclysm.eval.run_eval import make_policy  # noqa: E402
from cataclysm.policy import N_ACTIONS, OBS_DIM  # noqa: E402
from cataclysm.train.common import ActorCritic, StackBuffer, compute_gae, load_checkpoint, save_checkpoint  # noqa: E402
from cataclysm.train.ppo import load_train_config, train  # noqa: E402
from cataclysm.train.ppo_policy import PPOPolicy, load_policy  # noqa: E402


def test_stack_buffer_order_and_padding():
    s = StackBuffer(2, 3, 2)
    s.reset_rows(slice(None), np.array([[1, 1], [2, 2]], dtype=np.float32))
    assert s.flat()[0].tolist() == [0, 0, 0, 0, 1, 1]
    s.push(np.array([[3, 3], [4, 4]], dtype=np.float32))
    assert s.flat()[1].tolist() == [0, 0, 2, 2, 4, 4]
    s.reset_rows([0], np.array([9, 9], dtype=np.float32))   # скидання лише одного рядка
    assert s.flat()[0].tolist() == [0, 0, 0, 0, 9, 9]
    assert s.flat()[1].tolist() == [0, 0, 2, 2, 4, 4]


def test_gae_matches_manual():
    r = torch.tensor([[1.0], [1.0], [1.0]])
    v = torch.tensor([[0.5], [0.4], [0.3]])
    d = torch.tensor([[0.0], [1.0], [0.0]])      # епізод закінчується після кроку 1
    last = torch.tensor([0.2])
    adv, ret = compute_gae(r, v, d, last, gamma=0.9, lam=0.8)
    d2 = 1.0 + 0.9 * 0.2 - 0.3
    a2 = d2
    d1 = 1.0 - 0.4                                # кінець епізоду: без наступного значення
    a1 = d1
    d0 = 1.0 + 0.9 * 0.4 - 0.5
    a0 = d0 + 0.9 * 0.8 * a1
    assert adv[:, 0].tolist() == pytest.approx([a0, a1, a2], abs=1e-6)
    assert ret[:, 0].tolist() == pytest.approx([a0 + 0.5, a1 + 0.4, a2 + 0.3], abs=1e-6)


def test_policy_contract_and_checkpoint_roundtrip(tmp_path):
    net = ActorCritic(4 * OBS_DIM, hidden=[16], activation="tanh")
    save_checkpoint(tmp_path / "m.pt", net, 4, {"x": 1})
    net2, k, meta = load_checkpoint(tmp_path / "m.pt")
    assert k == 4 and meta == {"x": 1}
    cfg = load_env_config("L0")
    env = SwarmEnv(cfg)
    p1 = PPOPolicy(net, 4, deterministic=True)
    p2 = PPOPolicy(net2, 4, deterministic=True)
    for seed in (1, 2):
        r1, r2 = run_episode(env, p1, seed), run_episode(env, p2, seed)
        assert r1 == r2
    obs, _ = env.reset(seed=0)
    p1.reset()
    acts = p1.act(obs)
    assert set(acts) == set(obs) and all(0 <= a < N_ACTIONS for a in acts.values())


def test_stochastic_policy_is_seeded(tmp_path):
    net = ActorCritic(4 * OBS_DIM, hidden=[16])
    env = SwarmEnv(load_env_config("L0"))
    runs = [run_episode(env, PPOPolicy(net, 4, deterministic=False, seed=5), 3) for _ in range(2)]
    assert runs[0] == runs[1]


def _tiny_cfg():
    cfg = load_train_config()
    cfg.update({"n_envs": 2, "rollout_steps": 32, "minibatch": 32, "epochs": 2, "eval_every": 64,
                "eval_episodes": 2, "total_agent_steps": 256})
    cfg["net"] = {"type": "mlp", "hidden": [16], "activation": "tanh"}
    return cfg


def test_train_smoke_and_load_via_run_eval(tmp_path):
    run = train(_tiny_cfg(), seed=0, out_dir=tmp_path, levels=["L0", "L1"], total_steps=256, log=lambda *_: None)
    for f in ("best.pt", "last.pt", "stage_L0.pt", "stage_L1.pt", "train_log.csv", "eval_log.csv", "config.yaml"):
        assert (run / f).exists(), f
    cfg = load_env_config("L1")
    pol = make_policy(f"ppo:{run / 'best.pt'}", cfg)
    assert run_episode(SwarmEnv(cfg), pol, 10000)["steps"] > 0
    assert make_policy(f"ppo-det:{run}", cfg).deterministic   # каталог: бере best.pt


def test_train_is_reproducible(tmp_path):
    a = train(_tiny_cfg(), 3, tmp_path / "a", ["L0"], 128, log=lambda *_: None)
    b = train(_tiny_cfg(), 3, tmp_path / "b", ["L0"], 128, log=lambda *_: None)
    sa = torch.load(a / "last.pt", weights_only=True)["model"]
    sb = torch.load(b / "last.pt", weights_only=True)["model"]
    assert all(torch.equal(sa[k], sb[k]) for k in sa)


def test_load_policy_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_policy(tmp_path / "nope.pt")
