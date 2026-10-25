# SPEC: CATaclysm project contract

Version: **0.5** 

---

## 1. Goal and scope

Show that a decentralized swarm of two robots, each acting only on its own sensors, can reach a target (a light source), and that the trained policy is not worse than a hand-written baseline (ideally better on the harder levels). Then transfer the policy to the physical Codey Rocky robots.

Principle: **the agent sees only what the real Codey Rocky measures.** No global coordinates, no map, no position of the other robot.


## 2. Hardware and constraints

| Fact | Consequence |
|---|---|
| 2 robots, a PC, no other modules | N = 2 in all experiments |
| Arena measured: 200 x 200 cm (4 m2) | calibrated levels C0-C2 use 200 x 200 (idealized L0-L3 use 400 x 300) |
| No camera | orientation by light sensor: a bright flashlight is attached to the target |
| Light is measured by the sensor in the Codey head (scale 0-100) | one scalar, no direction; the agent finds the direction to the target by turning and comparing readings |
| The light channel of the front sensor on Rocky reads 100 at all distances with this flashlight | this channel is not used for light |
| Rocky has one IR sensor at the front; the API gives only the boolean `is_obstacle_ahead()` and a reflection scale 0-100, not centimeters | one beam `ir_front`, range about 30 cm |
| IR transmitter/receiver on Codey for robot-to-robot signalling | signal `mate_ir`: does the robot see the other one in front of it |
| Motors without encoders | motion is imprecise, noise is added in the simulation |
| The room must be dark | in a lit room the background is about 5 out of 100, which spoils the signal |

## 3. Observation (8 numbers per robot)

| Index | Field | Range | Meaning |
|---|---|---|---|
| 0 | `ir_front` | [0,1] | 1 - d/range for the forward IR beam, 0 if nothing is detected |
| 1 | `light` | [0,1] | normalized brightness: background + lamp contribution (sensor reading / 100) |
| 2 | `light_delta` | [-1,1] | change of `light` compared with the previous step, multiplied by `delta_scale` |
| 3 | `mate_ir` | {0,1} | whether the IR signal of the other robot is seen in front |
| 4-7 | `last_action` | one-hot | previous action (4 values) |

```yaml
obs_layout:
  - {name: ir_front, size: 1}
  - {name: light, size: 1}
  - {name: light_delta, size: 1}
  - {name: mate_ir, size: 1}
  - {name: last_action, size: 4}
```

The policy receives a stack of k = 4 frames (input size 32); at the start of an episode the older frames are zero. There are no recurrent networks (GRU).

## 4. Actions

Discrete, 4 values: `FORWARD=0`, `TURN_LEFT=1`, `TURN_RIGHT=2`, `STOP=3`. Step `dt = 0.2` s. Forward speed is 15 cm/s (measured). Turn rate is 150 deg/s (measured, about 30 degrees per step) in the calibrated levels C0-C2, and 90 deg/s in the idealized levels L0-L3.

```python
from typing import Protocol
import numpy as np

class Policy(Protocol):
    def reset(self) -> None: ...
    def act(self, obs: dict[str, np.ndarray]) -> dict[str, int]: ...
```

## 5. Light model

`reading = ambient + k * lobe(bearing) / (d^p + d0^p)`, where `lobe = max(cos, 0)^m`, multiplied by an occlusion factor (`occlusion_factor` when an obstacle is between the robot and the lamp), plus noise and quantization. The result is clipped to [0,1], so sensor saturation (reading 100) is part of the model.

Parameters of the calibrated levels C0-C2 (bright flashlight, head sensor, scale 0-1 = reading 0-100):

| Parameter | Value | Source |
|---|---|---|
| `k` | 14400 | signal is about 1.44e6 / d^2 on the 0-100 scale (d in cm), fitted on 150 and 175 cm |
| `exponent` p | 2.0 | the fit on 150 and 175 cm gave 1.94 |
| `d0_cm` | 5.0 | smoothing near the source |
| `lobe_exponent` m | 1.15 | from angles at 100 cm: 60 deg gives about 0.45 of the on-axis value, 90 deg gives 0; at 0, 30, 45 deg the sensor is saturated |
| `ambient` | 0.003 (C0, C1), 0.003-0.03 (C2) | dark room; the range for C2 is a margin for an imperfectly dark room |
| `sigma` | 0.004 | measured standard deviation 0.1-0.8 out of 100 |
| `quant_step` | 0.001 | reading resolution about 0.1 out of 100 |

Known discrepancy: the 130 cm point (98.8) lies above the pure 1/d^2 curve (it would be about 85), so the saturation is softer than a hard clip; the model has no soft saturation. The sensor saturates (100) up to about 120 cm on the axis.

From the side of the IR sensor the lamp looks like an obstacle (an object), so the baseline turns on AVOID only when `light < light_near`.

## 6. Difficulty levels

Idealized (used in development and in tests):

| Level | What changes |
|---|---|
| L0 | one target, one robot is enough for success, minimal noise |
| L1 | both robots must be at the target, variable background (0.02..0.20) |
| L2 | 3-6 random obstacles, light quantization, 5% loss of `mate_ir`, 5% motion noise |
| L3 | L2 + action delay of 1 step, stronger noise, 10% loss of `mate_ir` |

Calibrated on measurements:

| Level | What changes |
|---|---|
| C0 | arena 200 x 200, light and turn rate from measurements, one robot is enough |
| C1 | as C0, but both robots must reach the target |
| C2 | as C1 + 2-4 obstacles, motion noise (forward 20%, turning 30%, estimates), IR distance noise 5%, background 0.003-0.03 |

Configs: `config/env/base.yaml`, `L0..L3.yaml`, `C0..C2.yaml`. For L: arena 400 x 300, minimum start-to-target distance 150 cm, wall margin 20 cm. For C: arena 200 x 200, 100 cm and 15 cm. `max_steps = 500` and `capture_radius = 20` for all.

## 7. Reward

Step -0.01; shaping 0.1 for reducing the true distance to the target (training only, the agent does not see it); collision with an obstacle -0.2, with a robot -0.1; shared capture bonus 10.

## 8. Evaluation

- Episode seeds: training 0..8999, validation (checkpoint selection) 9000..9999, test 10000..10199 (200 episodes). The test seeds are not used in training. 3 training seeds (0, 1, 2).
- Metrics: `success_rate` (with a Wilson interval), `steps` (censored at max_steps), collisions, number of hunters at the moment of capture.
- CSV per episode: `policy, level, train_seed, episode_seed, success, steps, collisions_obstacle, collisions_agent, n_hunters_at_capture`.
- One command: `python -m cataclysm.eval.run_eval --policy greedy --level C1 --episodes 200 --out results/C1_greedy.csv`.
- Trained policy: `--policy ppo:models/ippo_seed0/best.pt --label ppo --train-seed 0` (`ppo-det:` takes the most probable action).
- Summary table: `python -m cataclysm.eval.table results/*.csv`.

### Training (IPPO, shared policy)

Command: `python -m cataclysm.train.ppo --seed N --levels C0,C1,C2` (hyperparameters in `config/train/ppo.yaml`). Curriculum C0, C1, C2 over 2 million agent steps, about 20 minutes per seed on a CPU. Network: MLP 64x64 tanh, separate actor and critic, both see only the local input. `best.pt` is selected by validation on the last level of the curriculum.

### Results on the calibrated levels C0-C2 (test, seeds 10000-10199)

| Policy | Level | Episodes | Success [95% CI] | Steps | Obstacle collisions | Robot-robot collisions |
|---|---|---|---|---|---|---|
| random | C0 | 200 | 0.080 [0.050, 0.126] | 486 | 36.0 | 3.4 |
| random | C1 | 200 | 0.000 [0.000, 0.019] | 500 | 34.2 | 2.1 |
| random | C2 | 200 | 0.000 [0.000, 0.019] | 500 | 38.4 | 3.0 |
| greedy | C0 | 200 | 0.970 [0.936, 0.986] | 144 | 0.1 | 7.9 |
| greedy | C1 | 200 | 0.755 [0.691, 0.809] | 245 | 0.6 | 30.4 |
| greedy | C2 | 200 | 0.665 [0.597, 0.727] | 347 | 30.1 | 11.8 |
| reactive | C0 | 200 | 0.975 [0.943, 0.989] | 142 | 0.1 | 6.4 |
| reactive | C1 | 200 | 0.770 [0.707, 0.823] | 242 | 0.6 | 27.6 |
| reactive | C2 | 200 | 0.710 [0.644, 0.768] | 340 | 33.0 | 8.1 |
| ppo, seed 0 | C0 / C1 / C2 | 200 | 1.000 / 1.000 / 1.000 | 55 / 73 / 93 | 0.0 / 0.0 / 5.3 | 0.8 / 2.0 / 2.1 |
| ppo, seed 1 | C0 / C1 / C2 | 200 | 1.000 / 1.000 / 1.000 | 58 / 82 / 110 | 0.0 / 0.0 / 9.3 | 0.9 / 5.6 / 2.4 |
| ppo, seed 2 | C0 / C1 / C2 | 200 | 1.000 / 1.000 / 0.995 | 56 / 73 / 106 | 0.0 / 0.0 / 6.9 | 0.6 / 3.4 / 2.2 |
| ppo, mean of 3 seeds | C0 | 600 | 1.000 +- 0.000 | 56 | 0.0 | 0.7 |
| ppo, mean of 3 seeds | C1 | 600 | 1.000 +- 0.000 | 76 | 0.0 | 3.7 |
| ppo, mean of 3 seeds | C2 | 600 | 0.998 +- 0.003 | 103 | 7.2 | 2.2 |

Conclusion: on C1 and C2 the trained policy is clearly better than the hand-written baselines (C1: 1.000 against 0.770, C2: 0.998 against 0.710), faster (C0: 56 against 142 steps), and collides less with obstacles and with the other robot. The three seeds give almost the same result. These are simulation results; the motion noise in C2 is an estimate, and transfer to the real robots is not verified.

### Historical results on the idealized levels L0-L2 (before calibration)

Baselines (200 test episodes):

| Policy | L0 | L1 | L2 |
|---|---|---|---|
| random | 0.040 [0.020, 0.077] | 0.000 [0.000, 0.019] | 0.000 [0.000, 0.019] |
| greedy (ScanAndGo) | 1.000 [0.981, 1.000] | 0.665 [0.597, 0.727] | 0.470 [0.402, 0.539] |
| reactive | 1.000 [0.981, 1.000] | 0.680 [0.612, 0.741] | 0.475 [0.407, 0.544] |

PPO (curriculum L0, L1, L2):

| Policy | L0 | L1 | L2 |
|---|---|---|---|
| ppo, seed 0 | 0.995 | 0.845 | 0.775 |
| ppo, seed 1 | 1.000 | 0.980 | 0.900 |
| ppo, seed 2 | 1.000 | 0.995 | 0.730 |
| ppo, mean of 3 seeds | 0.998 +- 0.003 | 0.940 +- 0.083 | 0.802 +- 0.088 |

These numbers are not used as the main results in the report, because the L environment does not match the real measurements.

## 9. Calibration on the real robot

Status of the measurements:

| # | What | Status |
|---|---|---|
| 1 | Light sensor: readings against distance (10-175 cm) and angle (0-90 deg at 100 cm), background | done with the bright flashlight; angles measured only at 100 cm, where the sensor is saturated at small angles; the angle series at 175 cm and the other side were not measured |
| 2 | Dynamic range, saturation, noise | done: saturation 100 up to about 120 cm, noise 0.1-0.8 out of 100 |
| 3 | Forward speed and angular speed | measured (15 cm/s, about 150 deg/s); repeats for the spread were not done, so the motion noise in C2 is an estimate |
| 4 | Arena area | done: 200 x 200 cm |

Raw measurements and the decoded CSV: `data/`. Scripts for the robot: `robot/`.

## 10. Repository structure

```
config/env/        base.yaml, L0..L3.yaml, C0..C2.yaml
config/train/      ppo.yaml
src/cataclysm/
  policy.py        policy contract
  env/             params.py, geometry.py, swarm_env.py
  baselines/       random_policy.py, scan_go.py
  eval/            metrics.py, run_eval.py, table.py
  viz/             render.py
  train/           common.py, ppo.py, ppo_policy.py, dummy_env.py
robot/             MicroPython scripts for Codey Rocky, decode_hex.py
data/              measurements on the real robots and data/README.md
models/            ippo_seed<N>/ (best.pt, last.pt, stage_*.pt, logs), not in git
tests/             test_env.py, test_eval.py, test_train.py
results/           CSV and GIF, not in git
```

Environment speed is about 1.6-2.1 thousand steps/s per core; target: at least 1500 steps/s.

## 11. Open questions

- [OPEN] Running the policy: on the PC with commands over Bluetooth/Wi-Fi, or on the robot itself; depends on the measured loop delay.
- [OPEN] Saturation of the head sensor up to about 120 cm: there is no light gradient inside this zone; the behaviour on the real robots must be checked (dim the flashlight if needed).
- Decided: arena 200 x 200 cm, bright flashlight, head sensor instead of the Rocky sensor, dark room.
- Decided: midterm on 26 Oct 2026 (a simulation and a justified behaviour are required; real robots by the final defense).

## 12. Changelog

- 0.5 (25 Oct 2026): measurements with the bright flashlight (1/d^2 law, cos^1.15 cone, saturation up to about 120 cm, noise), arena 200 x 200, turn rate 150 deg/s; new calibrated levels C0-C2 (`config/env/`); baseline and PPO results on C0-C2 (3 seeds); calibration status in section 9; L0-L3 stay as idealized levels.
- 0.4: clarified the observation layout (`obs_layout`), the results table and the calibration section; added robot scripts.
- 0.3 (18 Oct 2026): added IPPO (train/), train/val/test seed split, training results, removed GRU.
- 0.2 (11 Oct 2026): removed the camera; 8-dimensional observation with a light sensor; N = 2; arena 400 x 300; levels L0-L3; reference baseline results; calibration section.
- 0.1 (11 Oct 2026): first draft.