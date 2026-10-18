"""Заглушка середовища для відладки пайплайна навчання (Роль B) без залежності від env/.

Повертає випадкові observation тієї ж форми, що й SwarmEnv, і коротку «гру» з випадковим кінцем.
Лише для перевірки, що тренувальний код запускається, логує reward і зберігає/завантажує модель.
"""
from __future__ import annotations

import functools

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from cataclysm.policy import N_ACTIONS, OBS_DIM


class DummyEnv(ParallelEnv):
    metadata = {"name": "cataclysm_dummy_v0"}

    def __init__(self, n_hunters: int = 2, max_steps: int = 50):
        self.possible_agents = [f"hunter_{i}" for i in range(n_hunters)]
        self.agents: list[str] = []
        self.max_steps = max_steps
        self.t = 0
        self._rng = np.random.default_rng(0)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(-1.0, 1.0, (OBS_DIM,), np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(N_ACTIONS)

    def _obs(self):
        return {a: self._rng.uniform(-1, 1, OBS_DIM).astype(np.float32) for a in self.agents}

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        self.t = 0
        return self._obs(), {a: {} for a in self.agents}

    def step(self, actions):
        self.t += 1
        done = self.t >= self.max_steps
        obs = self._obs()
        rew = {a: float(self._rng.normal() * 0.1 + (1.0 if actions.get(a) == 0 else 0.0)) for a in self.agents}
        term = {a: False for a in self.agents}
        trunc = {a: done for a in self.agents}
        info = {a: {} for a in self.agents}
        if done:
            self.agents = []
        return obs, rew, term, trunc, info
