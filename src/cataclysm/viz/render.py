"""Візуалізація епізоду: кадр і GIF (SPEC, розділ 15). Роль A."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Circle, Rectangle, Wedge  # noqa: E402

HUNTER_COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b"]


def draw_frame(ax, data: dict, title: str | None = None) -> None:
    ax.clear()
    W, H = data["arena"]
    ax.add_patch(Rectangle((0, 0), W, H, fill=False, lw=2, ec="black"))
    for x, y, r in data["obstacles"]:
        ax.add_patch(Circle((x, y), r, fc="#888888", ec="black"))
    tx, ty = data["target"]
    ax.add_patch(Circle((tx, ty), data["capture_radius"], fc="none", ec="orange", ls="--", lw=1))
    ax.add_patch(Circle((tx, ty), data["target_radius"], fc="gold", ec="darkorange"))
    for i, ((x, y), h) in enumerate(zip(data["pos"], data["heading"])):
        c = HUNTER_COLORS[i % len(HUNTER_COLORS)]
        R = data["robot_radius"]
        ir = data["ir_range"]
        ax.add_patch(Wedge((x, y), R + ir, np.rad2deg(h) - 10, np.rad2deg(h) + 10, fc=c, alpha=0.12, ec="none"))
        ax.add_patch(Circle((x, y), R, fc=c, ec="black"))
        ax.plot([x, x + 1.8 * R * np.cos(h)], [y, y + 1.8 * R * np.sin(h)], color="black", lw=2)
        ax.text(x, y + R + 4, f"{data['light'][i]:.2f}", ha="center", fontsize=7, color=c)
    ax.set_xlim(-10, W + 10)
    ax.set_ylim(-10, H + 10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title or f"крок {data['t']}")


def rollout_gif(env, policy, path: str, seed: int = 0, fps: int = 10, stride: int = 2, max_frames: int = 250) -> int:
    """Проганяє один епізод і зберігає GIF. Повертає кількість кадрів."""
    obs, _ = env.reset(seed=seed)
    policy.reset()
    frames = [env.render_data()]
    captured = False
    while env.agents:
        obs, _, _, _, infos = env.step(policy.act(obs))
        captured = captured or any(i.get("captured") for i in infos.values())
        if env.t % stride == 0 or not env.agents:
            frames.append(env.render_data())
    frames = frames[:max_frames]
    fig, ax = plt.subplots(figsize=(6, 4.6), dpi=80)

    def update(k):
        draw_frame(ax, frames[k], f"крок {frames[k]['t']}" + ("  | ціль досягнута" if captured and k == len(frames) - 1 else ""))

    FuncAnimation(fig, update, frames=len(frames)).save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return len(frames)
