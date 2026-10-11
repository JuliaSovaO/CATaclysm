"""Спільний контракт між середовищем, baseline та навченими політиками (SPEC, розділ 9)."""
from typing import Protocol

import numpy as np

# Дії (SPEC, розділ 4)
FORWARD = 0
TURN_LEFT = 1
TURN_RIGHT = 2
STOP = 3
N_ACTIONS = 4

# Розкладка observation (SPEC, розділ 5). Один кадр = 8 чисел.
OBS_DIM = 8
OBS_IR_FRONT = 0
OBS_LIGHT = 1
OBS_LIGHT_DELTA = 2
OBS_MATE_IR = 3
OBS_LAST_ACTION = slice(4, 8)


class Policy(Protocol):
    def reset(self) -> None:
        """Викликається на початку епізоду: скидає внутрішню пам'ять (стек кадрів, стан автомата)."""

    def act(self, obs: dict[str, np.ndarray]) -> dict[str, int]:
        """obs: агент -> вектор розміру OBS_DIM (один кадр). Повертає агент -> дія 0..3.

        Політика бачить лише obs. Жодних infos, state() чи render_data().
        """
