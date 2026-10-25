"""2D-середовище рою CATaclysm (SPEC v0.2).

N роботів-переслідувачів (hunters) шукають лампу (ціль) лише за тим, що вміє Codey Rocky:
  - IR-дистанція вперед (один напрямок),
  - датчик світла (інтенсивність, без напрямку, але з діаграмою спрямованості),
  - IR-маяк товариша (чи бачу його спереду),
  - власна остання дія.
Глобальних координат агент не отримує.
"""
from __future__ import annotations

import functools

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from cataclysm.env.geometry import cast_rays, segment_blocked, wrap_angle
from cataclysm.policy import (
    FORWARD,
    N_ACTIONS,
    OBS_DIM,
    STOP,
    TURN_LEFT,
    TURN_RIGHT,
)


class SwarmEnv(ParallelEnv):
    metadata = {"name": "cataclysm_swarm_v0", "render_modes": []}

    def __init__(self, cfg: dict):
        self.cfg = cfg
        w, a, s = cfg["world"], cfg["agents"], cfg["sensors"]
        self.W, self.H = float(w["arena_size_cm"][0]), float(w["arena_size_cm"][1])
        self.dt = float(w["dt_s"])
        self.max_steps = int(w["max_steps"])
        self.n = int(a["n_hunters"])
        self.R = float(a["robot_radius_cm"])
        self.v = float(a["v_forward_cm_s"])
        self.omega = np.deg2rad(float(a["omega_deg_s"]))
        self.ir = s["ir"]
        self.light = s["light"]
        self.mate = s["mate_ir"]
        self.tcfg = cfg["target"]
        self.ocfg = cfg["obstacles"]
        self.spawn_cfg = cfg["spawn"]
        self.noise = cfg["noise"]
        self.rew = cfg["reward"]

        self.possible_agents = [f"hunter_{i}" for i in range(self.n)]
        self.agents: list[str] = []
        self._rng: np.random.Generator | None = None
        self._done = True

        # публічний стан (використовують viz і тести)
        self.t = 0
        self.pos = np.zeros((self.n, 2))
        self.heading = np.zeros(self.n)
        self.target = np.zeros(2)
        self.obstacles = np.zeros((0, 3))
        self.ambient = 0.0

    # ------------------------------------------------------------------ spaces
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Box:
        return spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Discrete:
        return spaces.Discrete(N_ACTIONS)

    # ------------------------------------------------------------------- reset
    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None or self._rng is None:
            self._rng = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        self._done = False
        self.t = 0
        self._spawn()

        lo, hi = self.light["ambient_range"]
        self.ambient = float(self._rng.uniform(lo, hi)) if hi > lo else float(lo)

        self._last_action = np.full(self.n, STOP, dtype=int)
        delay = int(self.noise["action_delay_steps"])
        self._queue = [np.full(self.n, STOP, dtype=int) for _ in range(delay)]
        self._t_heading = float(self._rng.uniform(-np.pi, np.pi))

        self._light_prev = self.light_readings(noise=True)
        self._light_now = self._light_prev.copy()
        obs = self._observe()
        infos = {a: {} for a in self.agents}
        return obs, infos

    # -------------------------------------------------------------------- step
    def step(self, actions: dict[str, int]):
        assert not self._done, "Епізод завершено, викличте reset()"
        agents = list(self.agents)
        act = np.array([int(actions.get(a, STOP)) for a in self.possible_agents], dtype=int)
        self._last_action = act.copy()
        if self._queue:
            self._queue.append(act)
            applied = self._queue.pop(0)
        else:
            applied = act

        prev_d = np.linalg.norm(self.pos - self.target, axis=1)
        hit_obs = np.zeros(self.n, dtype=bool)
        hit_agent = np.zeros(self.n, dtype=bool)

        for i in self._rng.permutation(self.n):
            a = applied[i]
            if a == FORWARD:
                length = self.v * self.dt * self._noisy(self.noise["move_sigma_frac"])
                direction = np.array([np.cos(self.heading[i]), np.sin(self.heading[i])])
                new = self.pos[i] + length * direction
                blocked_obs, blocked_agent, blocked_other = self._blocked(new, self.R, skip=i)
                if blocked_obs or blocked_agent or blocked_other:
                    hit_obs[i] = blocked_obs
                    hit_agent[i] = blocked_agent
                else:
                    self.pos[i] = new
            elif a == TURN_LEFT:
                self.heading[i] += self.omega * self.dt * self._noisy(self.noise["turn_sigma_frac"])
            elif a == TURN_RIGHT:
                self.heading[i] -= self.omega * self.dt * self._noisy(self.noise["turn_sigma_frac"])
            self.heading[i] = wrap_angle(self.heading[i])

        self._move_target()
        self.t += 1

        dists = np.linalg.norm(self.pos - self.target, axis=1)
        captured, n_near = self._check_capture(dists)

        shaping = self.rew["shaping"] * (prev_d - dists) / (self.v * self.dt)
        rewards_arr = (
            self.rew["step"]
            + shaping
            + self.rew["collision_obstacle"] * hit_obs
            + self.rew["collision_agent"] * hit_agent
        )
        if captured:
            rewards_arr = rewards_arr + self.rew["capture"]

        truncated = (not captured) and self.t >= self.max_steps
        self._light_prev = self._light_now
        obs = self._observe()

        rewards = {a: float(rewards_arr[self.possible_agents.index(a)]) for a in agents}
        terminations = {a: bool(captured) for a in agents}
        truncations = {a: bool(truncated) for a in agents}
        infos = {}
        for a in agents:
            i = self.possible_agents.index(a)
            infos[a] = {
                "collided_obstacle": bool(hit_obs[i]),
                "collided_agent": bool(hit_agent[i]),
            }
            if captured:
                infos[a]["captured"] = True
                infos[a]["n_hunters_at_capture"] = int(n_near)

        if captured or truncated:
            self._done = True
            self.agents = []
        return obs, rewards, terminations, truncations, infos

    # ------------------------------------------------------------ public views
    def state(self) -> np.ndarray:
        """Глобальний стан (лише для централізованого критика або дебагу)."""
        parts = [
            (self.pos / np.array([self.W, self.H])).ravel(),
            np.cos(self.heading),
            np.sin(self.heading),
            self.target / np.array([self.W, self.H]),
            [self.t / self.max_steps],
        ]
        return np.concatenate([np.asarray(p, dtype=np.float32).ravel() for p in parts])

    def render_data(self) -> dict:
        return {
            "arena": (self.W, self.H),
            "robot_radius": self.R,
            "pos": self.pos.copy(),
            "heading": self.heading.copy(),
            "target": self.target.copy(),
            "target_radius": float(self.tcfg["radius_cm"]),
            "capture_radius": float(self.tcfg["capture_radius_cm"]),
            "obstacles": self.obstacles.copy(),
            "t": self.t,
            "light": self._light_now.copy(),
            "ir_range": float(self.ir["range_cm"]),
        }

    # ---------------------------------------------------------------- sensors
    def _circles_for(self, i: int | None) -> np.ndarray:
        """Усі тверді кола для IR-променя: перешкоди, інші роботи, лампа."""
        rows = [self.obstacles]
        others = [j for j in range(self.n) if j != i]
        if others:
            rows.append(np.column_stack([self.pos[others], np.full(len(others), self.R)]))
        rows.append(np.array([[self.target[0], self.target[1], float(self.tcfg["radius_cm"])]]))
        return np.vstack(rows)

    def ir_readings(self, noise: bool = True) -> np.ndarray:
        h = np.deg2rad(float(self.ir["half_angle_deg"]))
        rng_cm = float(self.ir["range_cm"])
        out = np.ones(self.n)
        for i in range(self.n):
            hd = self.heading[i]
            ang = hd + np.array([-h, 0.0, h])
            dirs = np.column_stack([np.cos(ang), np.sin(ang)])
            origin = self.pos[i] + self.R * np.array([np.cos(hd), np.sin(hd)])
            d = cast_rays(origin, dirs, self._circles_for(i), self.W, self.H, rng_cm).min()
            if noise and self.noise["ir_dist_sigma_frac"] > 0:
                d *= 1.0 + self.noise["ir_dist_sigma_frac"] * self._rng.normal()
            out[i] = np.clip(d / rng_cm, 0.0, 1.0)
        return out

    def light_readings(self, noise: bool = True) -> np.ndarray:
        L = self.light
        vec = self.target - self.pos
        d = np.linalg.norm(vec, axis=1)
        phi = wrap_angle(np.arctan2(vec[:, 1], vec[:, 0]) - self.heading)
        m = float(L["lobe_exponent"])
        lobe = np.maximum(np.cos(phi), 0.0) ** m if m > 0 else np.ones(self.n)
        p = float(L["exponent"])
        inten = float(L["k"]) * lobe / (d**p + float(L["d0_cm"]) ** p)
        for i in range(self.n):
            if segment_blocked(self.pos[i], self.target, self.obstacles):
                inten[i] *= float(L["occlusion_factor"])
        reading = self.ambient + inten
        if noise and L["sigma"] > 0:
            reading = reading + float(L["sigma"]) * self._rng.normal(size=self.n)
        q = float(L["quant_step"])
        if q > 0:
            reading = np.round(reading / q) * q
        return np.clip(reading, 0.0, 1.0)

    def mate_ir_flags(self) -> np.ndarray:
        flags = np.zeros(self.n)
        M = self.mate
        if not M["enabled"] or self.n < 2:
            return flags
        rx = np.deg2rad(float(M["rx_half_angle_deg"]))
        tx = np.deg2rad(float(M["tx_half_angle_deg"]))
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                vij = self.pos[j] - self.pos[i]
                if np.linalg.norm(vij) > float(M["range_cm"]):
                    continue
                b_rx = wrap_angle(np.arctan2(vij[1], vij[0]) - self.heading[i])
                b_tx = wrap_angle(np.arctan2(-vij[1], -vij[0]) - self.heading[j])
                if abs(b_rx) > rx or abs(b_tx) > tx:
                    continue
                if segment_blocked(self.pos[i], self.pos[j], self.obstacles):
                    continue
                if float(M["dropout_p"]) > 0 and self._rng.random() < float(M["dropout_p"]):
                    continue
                flags[i] = 1.0
                break
        return flags

    def _observe(self) -> dict[str, np.ndarray]:
        ir = self.ir_readings()
        self._light_now = self.light_readings(noise=True)
        delta = np.clip((self._light_now - self._light_prev) * float(self.light["delta_scale"]), -1, 1)
        mate = self.mate_ir_flags()
        obs = {}
        for i, a in enumerate(self.possible_agents):
            v = np.zeros(OBS_DIM, dtype=np.float32)
            v[0] = ir[i]
            v[1] = self._light_now[i]
            v[2] = delta[i]
            v[3] = mate[i]
            v[4 + int(self._last_action[i])] = 1.0
            obs[a] = v
        return obs

    # ----------------------------------------------------------------- helpers
    def _noisy(self, sigma: float) -> float:
        return 1.0 + sigma * self._rng.normal() if sigma > 0 else 1.0

    def _blocked(self, new: np.ndarray, radius: float, skip: int | None) -> tuple[bool, bool, bool]:
        """(перешкода/стіна, інший робот, лампа) для диска радіуса radius у точці new."""
        m = radius
        wall = new[0] < m or new[0] > self.W - m or new[1] < m or new[1] > self.H - m
        obst = False
        if len(self.obstacles):
            obst = bool((np.linalg.norm(self.obstacles[:, :2] - new, axis=1) < self.obstacles[:, 2] + m).any())
        agent = False
        for j in range(self.n):
            if j != skip and np.linalg.norm(self.pos[j] - new) < 2 * self.R:
                agent = True
        lamp = bool(np.linalg.norm(self.target - new) < m + float(self.tcfg["radius_cm"]))
        return (wall or obst), agent, lamp

    def _check_capture(self, dists: np.ndarray) -> tuple[bool, int]:
        near = np.where(dists <= float(self.tcfg["capture_radius_cm"]))[0]
        n_near = len(near)
        if n_near < int(self.tcfg["capture_min_hunters"]):
            return False, n_near
        sep = float(self.tcfg["capture_min_separation_deg"])
        if sep > 0:
            ang = np.arctan2(self.pos[near, 1] - self.target[1], self.pos[near, 0] - self.target[0])
            best = 0.0
            for a in range(n_near):
                for b in range(a + 1, n_near):
                    best = max(best, abs(wrap_angle(ang[a] - ang[b])))
            if np.rad2deg(best) < sep:
                return False, n_near
        return True, n_near

    def _move_target(self) -> None:
        mode = self.tcfg["mode"]
        ratio = float(self.tcfg["speed_ratio"])
        if mode == "static" or ratio <= 0:
            return
        speed = ratio * self.v * self.dt
        direction = None
        if mode == "flee":
            vecs = self.target - self.pos
            dist = np.linalg.norm(vecs, axis=1)
            sensed = dist < float(self.tcfg["sense_radius_cm"])
            if sensed.any():
                push = (vecs[sensed] / (dist[sensed, None] ** 2)).sum(axis=0)
                tr = float(self.tcfg["radius_cm"])
                wall_w = float(self.tcfg["flee_weight_walls"])
                for axis, size in ((0, self.W), (1, self.H)):
                    if self.target[axis] < 2 * tr + 20:
                        push[axis] += wall_w
                    elif self.target[axis] > size - 2 * tr - 20:
                        push[axis] -= wall_w
                n = np.linalg.norm(push)
                if n > 1e-9:
                    direction = push / n
        if direction is None:
            self._t_heading += np.deg2rad(float(self.tcfg["random_turn_sigma_deg"])) * self._rng.normal()
            direction = np.array([np.cos(self._t_heading), np.sin(self._t_heading)])
        new = self.target + speed * direction
        tr = float(self.tcfg["radius_cm"])
        blocked_obs, _, _ = self._blocked(new, tr, skip=None)
        near_hunter = bool((np.linalg.norm(self.pos - new, axis=1) < self.R + tr).any())
        if blocked_obs or near_hunter:
            self._t_heading = float(self._rng.uniform(-np.pi, np.pi))
        else:
            self.target = new

    def _spawn(self) -> None:
        rng = self._rng
        margin = float(self.spawn_cfg["wall_margin_cm"])
        clr = float(self.ocfg["min_clearance_cm"])
        tr = float(self.tcfg["radius_cm"])

        def rand_point(m):
            return np.array([rng.uniform(m, self.W - m), rng.uniform(m, self.H - m)])

        # перешкоди
        obs = []
        n_max = int(self.ocfg["n_max"])
        if n_max > 0:
            count = int(rng.integers(int(self.ocfg["n_min"]), n_max + 1))
            r_lo, r_hi = self.ocfg["radius_cm"]
            for _ in range(count):
                for _try in range(100):
                    r = float(rng.uniform(r_lo, r_hi))
                    p = rand_point(r + margin)
                    if all(np.linalg.norm(p - o[:2]) > r + o[2] + clr for o in obs):
                        obs.append(np.array([p[0], p[1], r]))
                        break
        self.obstacles = np.array(obs).reshape(-1, 3)

        def free(p, radius):
            if len(self.obstacles) == 0:
                return True
            return bool((np.linalg.norm(self.obstacles[:, :2] - p, axis=1) > self.obstacles[:, 2] + radius + clr / 2).all())

        # ціль
        for _ in range(500):
            p = rand_point(margin + tr)
            if free(p, tr):
                self.target = p
                break
        else:
            self.target = np.array([self.W / 2, self.H / 2])

        # hunters
        min_d = float(self.spawn_cfg["min_hunter_target_dist_cm"])
        placed: list[np.ndarray] = []
        for i in range(self.n):
            attempts = 0
            while True:
                attempts += 1
                if attempts % 100 == 0:
                    min_d *= 0.8  # арена замала для заданої відстані, послаблюємо вимогу
                p = rand_point(margin)
                if np.linalg.norm(p - self.target) < min_d:
                    continue
                if not free(p, self.R + 5):
                    continue
                if any(np.linalg.norm(p - q) < 4 * self.R for q in placed):
                    continue
                placed.append(p)
                break
        self.pos = np.array(placed)
        self.heading = rng.uniform(-np.pi, np.pi, size=self.n)
