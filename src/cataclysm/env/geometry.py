"""Геометрія 2D: промені, перетини з колами й стінами, перекриття відрізком."""
from __future__ import annotations

import numpy as np


def wrap_angle(a):
    """Кут у діапазон [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def cast_rays(
    origin: np.ndarray,
    dirs: np.ndarray,
    circles: np.ndarray,
    width: float,
    height: float,
    max_range: float,
) -> np.ndarray:
    """Відстань уздовж кожного променя до найближчої стіни або кола (не більше max_range).

    origin: (2,), dirs: (R, 2) одиничні вектори, circles: (K, 3) = x, y, r.
    Якщо початок променя всередині кола, відстань 0.
    """
    t = np.full(len(dirs), float(max_range))

    for axis, size in ((0, width), (1, height)):
        d = dirs[:, axis]
        o = origin[axis]
        with np.errstate(divide="ignore", invalid="ignore"):
            tw = np.where(d > 1e-12, (size - o) / d, np.where(d < -1e-12, (0.0 - o) / d, np.inf))
        t = np.minimum(t, np.maximum(tw, 0.0))

    if len(circles):
        oc = circles[:, :2] - origin
        b = dirs @ oc.T
        c = (oc**2).sum(axis=1) - circles[:, 2] ** 2
        disc = b * b - c
        hit = disc >= 0
        sq = np.sqrt(np.where(hit, disc, 0.0))
        t1 = b - sq
        t2 = b + sq
        tc = np.where(t1 >= 0, t1, np.where(t2 >= 0, 0.0, np.inf))
        tc = np.where(hit, tc, np.inf)
        t = np.minimum(t, tc.min(axis=1))
    return t


def segment_blocked(p: np.ndarray, q: np.ndarray, circles: np.ndarray) -> bool:
    """True, якщо відрізок p-q перетинає хоч одне коло."""
    if len(circles) == 0:
        return False
    d = q - p
    l2 = float(d @ d)
    if l2 < 1e-12:
        return False
    c = circles[:, :2]
    t = np.clip(((c - p) @ d) / l2, 0.0, 1.0)
    closest = p + t[:, None] * d
    dist = np.linalg.norm(c - closest, axis=1)
    return bool((dist < circles[:, 2]).any())
