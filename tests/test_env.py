"""Приймальні тести середовища (SPEC, розділ 16)."""
import numpy as np
import pytest

from cataclysm.env import SwarmEnv, load_env_config
from cataclysm.policy import FORWARD, OBS_DIM, STOP, TURN_LEFT

QUIET = {"sensors": {"light": {"sigma": 0.0, "ambient_range": [0.05, 0.05]}}}


def make(level="L0", overrides=None):
    return SwarmEnv(load_env_config(level, overrides))


def random_episode(env, seed, rng):
    env.reset(seed=seed)
    while env.agents:
        yield env.step({a: int(rng.integers(4)) for a in env.agents})


@pytest.mark.parametrize("level", ["L0", "L1", "L2", "L3"])
def test_spaces_and_obs_ranges(level):
    env = make(level)
    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=0)
    for a in env.possible_agents:
        assert env.observation_space(a).shape == (OBS_DIM,)
        assert env.action_space(a).n == 4
        assert obs[a].shape == (OBS_DIM,) and obs[a].dtype == np.float32
    for ep in range(5):
        for o, *_ in random_episode(env, ep, rng):
            for v in o.values():
                assert np.isfinite(v).all()
                assert v.min() >= -1.0 and v.max() <= 1.0
                assert v[4:].sum() == 1.0  # one-hot останньої дії


def test_determinism():
    def trajectory(seed):
        env = make("L2")
        rng = np.random.default_rng(123)
        env.reset(seed=seed)
        out = []
        for _ in range(60):
            if not env.agents:
                break
            o, r, *_ = env.step({a: int(rng.integers(4)) for a in env.agents})
            out.append((env.pos.copy(), env.heading.copy(), {k: v.copy() for k, v in o.items()}, r))
        return out

    t1, t2 = trajectory(7), trajectory(7)
    assert len(t1) == len(t2)
    for (p1, h1, o1, r1), (p2, h2, o2, r2) in zip(t1, t2):
        assert np.array_equal(p1, p2) and np.array_equal(h1, h2) and r1 == r2
        for k in o1:
            assert np.array_equal(o1[k], o2[k])
    assert not np.array_equal(trajectory(7)[0][0], trajectory(8)[0][0])


def test_never_leaves_arena_or_overlaps_obstacles():
    env = make("L2")
    rng = np.random.default_rng(1)
    for ep in range(20):
        env.reset(seed=ep)
        while env.agents:
            env.step({a: int(rng.choice([FORWARD, FORWARD, TURN_LEFT])) for a in env.agents})
            R = env.R
            assert (env.pos[:, 0] >= R - 1e-9).all() and (env.pos[:, 0] <= env.W - R + 1e-9).all()
            assert (env.pos[:, 1] >= R - 1e-9).all() and (env.pos[:, 1] <= env.H - R + 1e-9).all()
            for x, y, r in env.obstacles:
                assert (np.linalg.norm(env.pos - [x, y], axis=1) >= r + R - 1e-9).all()
            if env.n > 1:
                assert np.linalg.norm(env.pos[0] - env.pos[1]) >= 2 * R - 1e-9


def test_100_random_episodes_finish():
    env = make("L1", {"world": {"max_steps": 100}})
    rng = np.random.default_rng(2)
    for ep in range(100):
        n = 0
        for _ in random_episode(env, ep, rng):
            n += 1
        assert n <= env.max_steps


def test_light_is_stronger_when_closer_and_when_facing_lamp():
    env = make("L0", QUIET)
    env.reset(seed=0)
    env.obstacles = np.zeros((0, 3))
    env.target = np.array([200.0, 150.0])
    env.pos[:] = [[50.0, 150.0], [350.0, 50.0]]
    env.heading[:] = [0.0, 0.0]  # перший дивиться на лампу
    facing = env.light_readings(noise=False)[0]
    env.heading[0] = np.pi  # відвернувся
    away = env.light_readings(noise=False)[0]
    assert facing > away
    env.heading[0] = 0.0
    env.pos[0] = [150.0, 150.0]  # ближче
    closer = env.light_readings(noise=False)[0]
    assert closer > facing


def test_light_occlusion_by_obstacle():
    env = make("L0", QUIET)
    env.reset(seed=0)
    env.target = np.array([200.0, 150.0])
    env.pos[0] = [100.0, 150.0]
    env.heading[0] = 0.0
    env.obstacles = np.zeros((0, 3))
    open_view = env.light_readings(noise=False)[0]
    env.obstacles = np.array([[150.0, 150.0, 8.0]])
    blocked = env.light_readings(noise=False)[0]
    assert blocked < open_view


def test_ir_sees_obstacle_and_wall():
    env = make("L0", QUIET)
    env.reset(seed=0)
    env.obstacles = np.zeros((0, 3))
    env.target = np.array([300.0, 250.0])  # далеко від променя
    env.pos[0] = [100.0, 150.0]
    env.heading[0] = 0.0
    assert env.ir_readings(noise=False)[0] == pytest.approx(1.0)
    env.obstacles = np.array([[100.0 + env.R + 15.0 + 5.0, 150.0, 5.0]])  # поверхня за 15 см від краю робота
    assert env.ir_readings(noise=False)[0] == pytest.approx(15.0 / env.ir["range_cm"], abs=1e-6)
    env.obstacles = np.zeros((0, 3))
    env.pos[0] = [env.W - env.R - 10.0, 150.0]  # стіна за 10 см
    assert env.ir_readings(noise=False)[0] == pytest.approx(10.0 / env.ir["range_cm"], abs=1e-6)


def test_capture_terminates_episode_and_rewards():
    env = make("L0")
    env.reset(seed=0)
    env.pos[0] = env.target + [15.0, 0.0]  # у радіусі ловлі, поза лампою
    obs, rew, term, trunc, info = env.step({a: STOP for a in env.agents})
    assert all(term.values()) and not any(trunc.values())
    assert info["hunter_0"]["captured"] and info["hunter_0"]["n_hunters_at_capture"] >= 1
    assert rew["hunter_0"] > 5  # бонус за ловлю
    assert env.agents == []
    with pytest.raises(AssertionError):
        env.step({})


def test_capture_requires_both_hunters_on_L1():
    env = make("L1")
    env.reset(seed=0)
    env.pos[0] = env.target + [15.0, 0.0]
    _, _, term, _, _ = env.step({a: STOP for a in env.agents})
    assert not any(term.values())  # поруч лише один
    env.pos[1] = env.target + [-15.0, 0.0]
    _, _, term, _, info = env.step({a: STOP for a in env.agents})
    assert all(term.values()) and info["hunter_0"]["n_hunters_at_capture"] == 2


def test_truncation_at_max_steps():
    env = make("L0", {"world": {"max_steps": 5}})
    env.reset(seed=0)
    n = 0
    while env.agents:
        _, _, term, trunc, _ = env.step({a: STOP for a in env.agents})
        n += 1
    assert n == 5 and all(trunc.values()) and not any(term.values())


def test_action_delay_applies_previous_action():
    env = make("L0", {"noise": {"action_delay_steps": 1}})
    env.reset(seed=0)
    h0 = env.heading.copy()
    env.step({a: TURN_LEFT for a in env.agents})  # перший крок: у черзі STOP
    assert np.allclose(env.heading, h0)
    env.step({a: STOP for a in env.agents})  # тепер виконується відкладений TURN_LEFT
    assert not np.allclose(env.heading, h0)
