"""Завантаження конфігів середовища: base.yaml + рівень (L0..L3) + ручні перекриття."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

_DEFAULT_DIR = Path(__file__).resolve().parents[3] / "config" / "env"


def _deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_env_config(
    level: str | None = None,
    overrides: dict | None = None,
    config_dir: str | Path | None = None,
) -> dict:
    """Повертає повну конфігурацію середовища.

    level: "L0".."L3" або None (тільки base.yaml).
    overrides: вкладений словник, що перекриває будь-які ключі (для тестів і експериментів).
    """
    cdir = Path(config_dir) if config_dir else _DEFAULT_DIR
    with open(cdir / "base.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if level:
        path = cdir / f"{level}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Немає конфігу рівня {level}: {path}")
        with open(path, encoding="utf-8") as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg
