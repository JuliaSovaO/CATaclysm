"""Нижня межа: рівномірно випадкові дії."""
import numpy as np

from cataclysm.policy import N_ACTIONS


class RandomPolicy:
    def __init__(self, seed: int = 0):
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass  # потік випадкових чисел не скидаємо: епізоди мають різні дії

    def act(self, obs):
        return {a: int(self._rng.integers(N_ACTIONS)) for a in obs}
