"""IPPO зі спільною політикою (parameter sharing) для рою з двох роботів.

Кожен робот у кожному середовищі це окремий потік досвіду; всі потоки навчають одну мережу.
Вхід політики: стек із k останніх локальних спостережень (SPEC, розділ 3). Критик теж локальний.

Запуск:
    python -m cataclysm.train.ppo --config config/train/ppo.yaml --seed 0
    python -m cataclysm.train.ppo --levels L1 --total-steps 500000 --seed 1   # одна ступінь
Результат: models/ippo_seed<seed>/{best.pt, last.pt, stage_<рівень>.pt, train_log.csv, eval_log.csv, config.yaml}
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.distributions import Categorical

from cataclysm.env import SwarmEnv, load_env_config
from cataclysm.eval.metrics import run_episode, summarize
from cataclysm.policy import OBS_DIM
from cataclysm.train.common import ActorCritic, StackBuffer, compute_gae, save_checkpoint
from cataclysm.train.ppo_policy import PPOPolicy

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_SEED_MAX = 9000     # тренувальні епізоди: seed 0..8999
VAL_SEED0 = 9000          # валідація: 9000..9999 (відбір чекпоінта); тест 10000.. не чіпаємо

TRAIN_COLS = ["agent_steps", "stage", "level", "ep_return", "ep_success", "ep_len", "episodes",
              "entropy", "pg_loss", "v_loss", "approx_kl", "clipfrac", "expl_var", "lr", "steps_per_s"]
EVAL_COLS = ["agent_steps", "stage", "level", "success_rate", "ci_low", "ci_high", "steps_mean_all",
             "collisions_obstacle", "collisions_agent", "is_best"]


def load_train_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else REPO_ROOT / "config" / "train" / "ppo.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _stack_obs(env: SwarmEnv, obs: dict) -> np.ndarray:
    return np.stack([obs[a] for a in env.possible_agents]).astype(np.float32)


def validate(net: ActorCritic, env_cfg: dict, frame_stack: int, episodes: int, deterministic: bool) -> dict:
    env = SwarmEnv(env_cfg)
    policy = PPOPolicy(net, frame_stack, deterministic=deterministic, seed=0)
    rows = [run_episode(env, policy, VAL_SEED0 + i) for i in range(episodes)]
    return summarize(rows)


class _CsvLog:
    def __init__(self, path: Path, cols: list[str]):
        self.cols = cols
        self.f = open(path, "w", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=cols)
        self.w.writeheader()

    def write(self, row: dict) -> None:
        self.w.writerow({c: row.get(c, "") for c in self.cols})
        self.f.flush()

    def close(self) -> None:
        self.f.close()


def train(cfg: dict, seed: int, out_dir: str | Path = "models", levels: list[str] | None = None,
          total_steps: int | None = None, log=print) -> Path:
    """Навчає політику й повертає каталог запуску."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    seed_rng = np.random.default_rng(seed)
    torch_gen = torch.Generator().manual_seed(seed)

    levels = list(levels or cfg["curriculum"])
    total = int(total_steps or cfg["total_agent_steps"])
    k = int(cfg["frame_stack"])
    n_envs, T = int(cfg["n_envs"]), int(cfg["rollout_steps"])
    gamma, lam, clip = float(cfg["gamma"]), float(cfg["gae_lambda"]), float(cfg["clip"])
    net_cfg = cfg["net"]
    if net_cfg.get("type", "mlp") != "mlp":
        raise ValueError("Підтримується лише net.type = mlp")

    n_agents = int(load_env_config(levels[0])["agents"]["n_hunters"])
    N = n_envs * n_agents
    F = k * OBS_DIM
    per_update = T * N
    updates_per_stage = max(1, math.ceil(total / len(levels) / per_update))
    total_updates = updates_per_stage * len(levels)

    net = ActorCritic(F, hidden=net_cfg["hidden"], activation=net_cfg["activation"])
    opt = torch.optim.Adam(net.parameters(), lr=float(cfg["lr"]), eps=1e-5)

    run_dir = Path(out_dir) / f"ippo_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({**cfg, "seed": seed, "levels": levels, "total_agent_steps": total}, f, allow_unicode=True)
    train_log = _CsvLog(run_dir / "train_log.csv", TRAIN_COLS)
    eval_log = _CsvLog(run_dir / "eval_log.csv", EVAL_COLS)

    eval_every = int(cfg["eval_every"])
    eval_eps = int(cfg["eval_episodes"])
    eval_det = bool(cfg.get("eval_deterministic", False))
    best_key = (-1.0, 0.0)
    last_eval = -1
    agent_steps = 0
    next_eval = eval_every
    upd_global = 0
    t0 = time.time()

    def do_eval(level: str, env_cfg: dict, stage: int, final_stage: bool) -> None:
        nonlocal best_key, last_eval
        if agent_steps == last_eval:
            return
        last_eval = agent_steps
        s = validate(net, env_cfg, k, eval_eps, eval_det)
        key = (s["success_rate"], -s["steps_mean_all"])
        is_best = final_stage and key > best_key
        if is_best:
            best_key = key
            save_checkpoint(run_dir / "best.pt", net, k, {"agent_steps": agent_steps, "level": level, "seed": seed})
        eval_log.write({"agent_steps": agent_steps, "stage": stage, "level": level,
                        "success_rate": s["success_rate"], "ci_low": s["success_ci_low"],
                        "ci_high": s["success_ci_high"], "steps_mean_all": s["steps_mean_all"],
                        "collisions_obstacle": s["collisions_obstacle_mean"],
                        "collisions_agent": s["collisions_agent_mean"], "is_best": int(is_best)})
        log(f"[eval] {level} кроків {agent_steps}: success {s['success_rate']:.3f} "
            f"[{s['success_ci_low']:.2f},{s['success_ci_high']:.2f}] steps {s['steps_mean_all']:.0f}"
            f"{'  *best*' if is_best else ''}")

    for si, level in enumerate(levels):
        env_cfg = load_env_config(level)
        final_stage = si == len(levels) - 1
        envs = [SwarmEnv(env_cfg) for _ in range(n_envs)]
        stack = StackBuffer(N, k, OBS_DIM)
        ep_ret = np.zeros(n_envs)
        ep_len = np.zeros(n_envs, dtype=int)

        def new_seed() -> int:
            return int(seed_rng.integers(0, TRAIN_SEED_MAX))

        def rows_of(e: int) -> slice:
            return slice(e * n_agents, (e + 1) * n_agents)

        for e, env in enumerate(envs):
            o, _ = env.reset(seed=new_seed())
            stack.reset_rows(rows_of(e), _stack_obs(env, o))

        for _ in range(updates_per_stage):
            lr = float(cfg["lr"]) * (1.0 - 0.9 * upd_global / total_updates)
            for g in opt.param_groups:
                g["lr"] = lr
            buf_x = torch.zeros(T, N, F)
            buf_a = torch.zeros(T, N, dtype=torch.long)
            buf_lp = torch.zeros(T, N)
            buf_v = torch.zeros(T, N)
            buf_r = torch.zeros(T, N)
            buf_d = torch.zeros(T, N)
            finished: list[tuple[float, int, int]] = []
            t_roll = time.time()

            for t in range(T):
                x = torch.from_numpy(stack.flat())
                with torch.no_grad():
                    logits, v = net(x)
                    dist = Categorical(logits=logits)
                    a = dist.sample()
                    lp = dist.log_prob(a)
                buf_x[t], buf_a[t], buf_lp[t], buf_v[t] = x, a, lp, v
                a_np = a.numpy().reshape(n_envs, n_agents)

                rew = np.zeros(N, dtype=np.float32)
                done = np.zeros(N, dtype=np.float32)
                new_obs = np.zeros((N, OBS_DIM), dtype=np.float32)
                ended: list[int] = []
                truncated_rows: list[int] = []
                for e, env in enumerate(envs):
                    names = env.possible_agents
                    actions = {ag: int(a_np[e, i]) for i, ag in enumerate(names)}
                    o, r, term, trunc, info = env.step(actions)
                    sl = rows_of(e)
                    new_obs[sl] = _stack_obs(env, o)
                    rew[sl] = [r[ag] for ag in names]
                    ep_ret[e] += float(rew[sl].mean())
                    ep_len[e] += 1
                    if not env.agents:
                        ended.append(e)
                        done[sl] = 1.0
                        success = any(info[ag].get("captured", False) for ag in names)
                        finished.append((ep_ret[e], int(success), int(ep_len[e])))
                        if not success:   # обрізання за часом: бутстрап значення кінцевого стану
                            truncated_rows += list(range(sl.start, sl.stop))
                stack.push(new_obs)
                if truncated_rows:
                    with torch.no_grad():
                        vb = net.value(torch.from_numpy(stack.flat()[truncated_rows])).numpy()
                    rew[truncated_rows] += gamma * vb
                for e in ended:
                    o, _ = envs[e].reset(seed=new_seed())
                    stack.reset_rows(rows_of(e), _stack_obs(envs[e], o))
                    ep_ret[e], ep_len[e] = 0.0, 0
                buf_r[t] = torch.from_numpy(rew)
                buf_d[t] = torch.from_numpy(done)

            with torch.no_grad():
                last_v = net.value(torch.from_numpy(stack.flat()))
            adv, ret = compute_gae(buf_r, buf_v, buf_d, last_v, gamma, lam)

            b_x, b_a, b_lp = buf_x.reshape(-1, F), buf_a.reshape(-1), buf_lp.reshape(-1)
            b_adv, b_ret, b_v = adv.reshape(-1), ret.reshape(-1), buf_v.reshape(-1)
            B = b_x.shape[0]
            mb = int(cfg["minibatch"])
            stats = {"pg": [], "vl": [], "ent": [], "kl": [], "cf": []}
            stop = False
            for _epoch in range(int(cfg["epochs"])):
                idx = torch.randperm(B, generator=torch_gen)
                for start in range(0, B, mb):
                    j = idx[start:start + mb]
                    logits, v = net(b_x[j])
                    dist = Categorical(logits=logits)
                    new_lp = dist.log_prob(b_a[j])
                    ent = dist.entropy().mean()
                    log_ratio = new_lp - b_lp[j]
                    ratio = log_ratio.exp()
                    adv_j = b_adv[j]
                    adv_j = (adv_j - adv_j.mean()) / (adv_j.std() + 1e-8)
                    pg = -torch.min(ratio * adv_j, ratio.clamp(1 - clip, 1 + clip) * adv_j).mean()
                    vl = 0.5 * ((v - b_ret[j]) ** 2).mean()
                    loss = pg + float(cfg["vf_coef"]) * vl - float(cfg["ent_coef"]) * ent
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), float(cfg["max_grad_norm"]))
                    opt.step()
                    with torch.no_grad():
                        kl = ((ratio - 1) - log_ratio).mean().item()
                        stats["pg"].append(pg.item())
                        stats["vl"].append(vl.item())
                        stats["ent"].append(ent.item())
                        stats["kl"].append(kl)
                        stats["cf"].append(((ratio - 1).abs() > clip).float().mean().item())
                    if cfg.get("target_kl") and kl > float(cfg["target_kl"]):
                        stop = True
                        break
                if stop:
                    break

            var_y = b_ret.var().item()
            expl_var = float("nan") if var_y < 1e-12 else 1.0 - (b_ret - b_v).var().item() / var_y
            agent_steps += per_update
            upd_global += 1
            n_fin = len(finished)
            train_log.write({
                "agent_steps": agent_steps, "stage": si, "level": level,
                "ep_return": np.mean([f[0] for f in finished]) if n_fin else "",
                "ep_success": np.mean([f[1] for f in finished]) if n_fin else "",
                "ep_len": np.mean([f[2] for f in finished]) if n_fin else "",
                "episodes": n_fin, "entropy": np.mean(stats["ent"]), "pg_loss": np.mean(stats["pg"]),
                "v_loss": np.mean(stats["vl"]), "approx_kl": np.mean(stats["kl"]),
                "clipfrac": np.mean(stats["cf"]), "expl_var": expl_var, "lr": lr,
                "steps_per_s": per_update / (time.time() - t_roll)})
            if upd_global % 10 == 0 or upd_global == total_updates:
                log(f"[{level}] {agent_steps}/{updates_per_stage * len(levels) * per_update} "
                    f"succ {np.mean([f[1] for f in finished]) if n_fin else float('nan'):.2f} "
                    f"ent {np.mean(stats['ent']):.3f} kl {np.mean(stats['kl']):.4f} "
                    f"ev {expl_var:.2f} ({time.time() - t0:.0f}s)")
            if agent_steps >= next_eval:
                do_eval(level, env_cfg, si, final_stage)
                next_eval += eval_every

        do_eval(level, env_cfg, si, final_stage)   # в кінці кожної ступені
        save_checkpoint(run_dir / f"stage_{level}.pt", net, k, {"agent_steps": agent_steps, "level": level, "seed": seed})

    save_checkpoint(run_dir / "last.pt", net, k, {"agent_steps": agent_steps, "level": levels[-1], "seed": seed})
    if not (run_dir / "best.pt").exists():
        save_checkpoint(run_dir / "best.pt", net, k, {"agent_steps": agent_steps, "level": levels[-1], "seed": seed})
    train_log.close()
    eval_log.close()
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=None, help="за замовчуванням перший з seeds у конфігу")
    ap.add_argument("--levels", default=None, help="напр. L0,L1,L2 (за замовчуванням curriculum з конфігу)")
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--out", default=None, help="за замовчуванням checkpoint_dir з конфігу")
    args = ap.parse_args()

    cfg = load_train_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg["seeds"][0])
    levels = args.levels.split(",") if args.levels else None
    run_dir = train(cfg, seed, args.out or cfg["checkpoint_dir"], levels, args.total_steps)
    print(f"Готово: {run_dir}")


if __name__ == "__main__":
    main()
