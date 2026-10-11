"""Baseline без навчання за датчиком світла: «скануй, вирівняйся, їдь» (SPEC, розділ 13).

Кожен робот діє незалежно і бачить лише власне observation:
  SCAN  : повний оберт на місці, запам'ятовуємо яскравість у кожному положенні;
  ALIGN : повертаємо до положення з максимальною яскравістю;
  GO    : їдемо вперед (чим темніше, тим довше), потім знову SCAN;
  AVOID : рефлекс уникнення, якщо IR бачить перешкоду, а світло ще не яскраве (інакше це, ймовірно, лампа).

separate=True додає правило відсторонення від товариша (mate_ir), це і є ReactiveRules (Boids-стиль).
"""
from __future__ import annotations

import numpy as np

from cataclysm.policy import (
    FORWARD,
    OBS_IR_FRONT,
    OBS_LIGHT,
    OBS_MATE_IR,
    TURN_LEFT,
    TURN_RIGHT,
)


class ScanAndGo:
    def __init__(
        self,
        scan_steps: int = 20,
        ir_thresh: float = 0.2,
        light_near: float = 0.45,
        avoid_steps: int = 3,
        go_min: int = 5,
        go_max: int = 60,
        min_contrast: float = 0.01,
        separate: bool = False,
        seed: int = 0,
    ):
        self.S = int(scan_steps)
        self.ir_thresh = ir_thresh
        self.light_near = light_near
        self.avoid_steps = avoid_steps
        self.go_min, self.go_max = go_min, go_max
        self.min_contrast = min_contrast
        self.separate = separate
        self._rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        self._st: dict[str, dict] = {}

    def act(self, obs):
        return {a: self._act_one(a, o) for a, o in obs.items()}

    # ------------------------------------------------------------------
    def _new_state(self) -> dict:
        return {"mode": "SCAN", "scan_i": 0, "readings": [], "align": 0, "go_left": 0, "avoid_left": 0}

    def _act_one(self, agent: str, o: np.ndarray) -> int:
        st = self._st.setdefault(agent, self._new_state())
        light = float(o[OBS_LIGHT])

        # рефлекси: перешкода спереду (але не лампа) або товариш спереду
        obstacle = o[OBS_IR_FRONT] < self.ir_thresh and light < self.light_near
        mate = self.separate and o[OBS_MATE_IR] > 0.5 and o[OBS_IR_FRONT] < 0.8
        if st["mode"] in ("GO", "ALIGN") and (obstacle or mate):
            st["mode"] = "AVOID"
            st["avoid_left"] = self.avoid_steps + (2 if mate else 0)
            st["avoid_dir"] = TURN_RIGHT if self._rng.random() < 0.5 else TURN_LEFT

        if st["mode"] == "AVOID":
            if st["avoid_left"] > 0:
                st["avoid_left"] -= 1
                return st["avoid_dir"]
            st.update(self._new_state())

        if st["mode"] == "SCAN":
            if st["scan_i"] == 0:
                st["readings"] = [light]
            else:
                st["readings"].append(light)
            if st["scan_i"] < self.S:
                st["scan_i"] += 1
                return TURN_LEFT
            r = np.array(st["readings"][: self.S])
            smooth = (np.roll(r, 1) + r + np.roll(r, -1)) / 3.0
            if r.max() - r.min() < self.min_contrast:
                st["align"] = int(self._rng.integers(0, self.S))  # сигналу не видно: випадковий напрямок
            else:
                st["align"] = int(np.argmax(smooth))
            st["mode"] = "ALIGN"

        if st["mode"] == "ALIGN":
            if st["align"] > 0:
                st["align"] -= 1
                return TURN_LEFT
            frac = min(light, 0.6) / 0.6
            st["go_left"] = int(np.clip(self.go_max * (1.0 - frac), self.go_min, self.go_max))
            st["mode"] = "GO"

        # GO
        if st["go_left"] > 0:
            st["go_left"] -= 1
            return FORWARD
        st.update(self._new_state())
        return self._act_one(agent, o)
