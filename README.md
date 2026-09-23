# ParkingAgent — Autonomous Car Parking with Reinforcement Learning

A Unity ML-Agents project in which a car learns to park itself between two parked cars using Proximal Policy Optimization.

Author: Kanan Sofiyev
---

## Overview

The agent controls a car in a 3D parking lot. In every episode the car starts at a random position with a random heading and has to drive into a marked parking spot, align with it and stop, without hitting the walls or the neighbouring parked cars.

| | |
|---|---|
| Engine | Unity 6 (6000.6), Universal Render Pipeline |
| ML toolkit | Unity ML-Agents (Unity package 4.x, Python `mlagents` 1.1.0) |
| Algorithm | PPO |
| Environment | 3D, 9 parallel training areas |
| Actions | 3 continuous actions in [-1, 1] |
| Sensors | RayPerceptionSensor3D + vector observations |

---

## Environment

Each training area (30 × 30) contains:

- a ground plane surrounded by four walls (tag `Wall`),
- a green parking spot,
- two parked cars on both sides of the spot (tag `Obstacle`),
- the agent car (blue, with a yellow marker on its front).

**Random components:** at the start of every episode the car is placed at a random position in the lower half of the area with a random rotation of ±60° and because of it the agent cannot memorise a single trajectory.

**Parallel environments:** the scene contains 9 copies of the training area arranged in a 3×3 grid. All 9 agents share one policy and collect experience simultaneously, which speeds up training roughly 9×.

---

## Agent

### Observations

Vector observations (5 values):

1. direction to the parking spot in the car's local frame, x component (normalised),
2. direction to the parking spot in the car's local frame, z component (normalised),
3. dot product of the car's forward vector and the spot's forward vector (alignment),
4. dot product of the car's right vector and the spot's forward vector (alignment sign),
5. current speed / max speed.

Additionally, a **RayPerceptionSensor3D** casts 13 rays over 360° (6 per direction, max 180°, length 15 m) and detects objects tagged `Wall` and `Obstacle`, so the agent can see the obstacles around it, including behind the car if reversing.

### Actions (3-dimensional)

| Index | Meaning | Range |
|---|---|---|
| 0 | steering | -1 (left) … 1 (right) |
| 1 | throttle | -1 (reverse) … 1 (forward) |
| 2 | brake | active only above 0.5, strength scaled to 0…1 |

The brake uses a threshold so that random actions at the beginning of training do not keep the car braking all the time. Without this the agent barely moved and could not explore (see *Lessons learned*).

Steering is scaled by the current speed, so the car cannot rotate in place, like a real car.

### Rewards

| Event | Reward |
|---|---|
| every step | −1 / MaxStep (time penalty, −1 in total if the episode times out) |
| getting closer to the spot | +0.1 per metre of distance reduced (−0.1 per metre moved away) |
| successful parking | +2, episode ends |
| collision with a wall or a parked car | −1, episode ends |

Parking counts as successful when the car is within 1 m of the spot's centre, is aligned within 15° (forwards or backwards) and its speed is below 0.3 m/s.

Episode length: `MaxStep = 2000` physics steps, `Decision Period = 5`, i.e. at most 400 decisions per episode.

---

## Training

Training was done with a standalone Windows build of the environment, launched by `mlagents-learn` without graphics, on the CPU.

```
mlagents-learn config/car_parking.yaml --run-id=parking_v5 --env=Build/ParkingAgent.exe --no-graphics --timeout-wait=300
```

### Baseline configuration (`config/car_parking.yaml`)

| Parameter | Value |
|---|---|
| trainer | PPO |
| batch_size / buffer_size | 1024 / 10240 |
| learning_rate | 3e-4 (linear schedule) |
| beta / epsilon / lambda | 5e-3 / 0.2 / 0.95 |
| num_epoch | 3 |
| network | 2 hidden layers × 256 units, input normalisation |
| gamma | 0.99 |
| time_horizon | 128 |
| max_steps | 2,000,000 |

Training progress was monitored with **TensorBoard**:

```
tensorboard --logdir results
```

---

## Results

### Baseline (`parking_v5`)

- Mean cumulative reward rose from **−1.0** to about **+3.3** within 600k steps and reached a plateau of **≈3.67** at 2M steps.
- Mean episode length dropped from ~380 decisions (time-outs) to **~60 decisions**: the agent parks quickly and consistently.
- Full training (2M steps, 9 parallel areas) took about 27 minutes.

### Hyperparameter comparison

Two additional runs change a single parameter compared to the baseline (1M steps each):

| Run | Config | Change | Result |
|---|---|---|---|
| `parking_v5` | `car_parking.yaml` | baseline (2×256, lr 3e-4) | plateau ≈ 3.67 |
| `parking_small` | `car_parking_small.yaml` | smaller network: 2×64 | *to be filled in* |
| `parking_lr` | `car_parking_lr.yaml` | higher learning rate: 1e-3 | *to be filled in* |

The runs can be compared in TensorBoard by selecting all three in the left panel.

---

## How to run

### Requirements

- Unity 6 (6000.6) with the ML Agents package
- Python **3.10.11**
- `torch==2.2.1`, `mlagents==1.1.0`

```
py -3.10 -m venv venv
venv\Scripts\activate
pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121
pip install mlagents==1.1.0
```

### Watch the trained agent

Open `Assets/Scenes/SampleScene`, select the `Car` objects, set **Behavior Parameters → Model** to `Assets/Models/CarParking.onnx` and **Behavior Type** to Inference Only, then press play.

### Train from scratch

1. Set **Behavior Type** of all cars to `Default`.
2. Build the scene into the `Build` folder.
3. Run:

```
venv\Scripts\activate
set CUDA_VISIBLE_DEVICES=-1
mlagents-learn config/car_parking.yaml --run-id=my_run --env=Build/ParkingAgent.exe --no-graphics --timeout-wait=300
```

### Known issues on Windows

- `TypeError: Invalid first argument to register()` — the old `cattrs` library used by `mlagents` is incompatible with Python ≥ 3.10.2. Fixed by a one-line patch in `venv/Lib/site-packages/cattr/dispatch.py`.
- `Expected all tensors to be on the same device` with `--torch-device cpu` — hide the GPU instead with `set CUDA_VISIBLE_DEVICES=-1`.
- Training inside the Unity 6.6 Editor froze after the first actions were sent, so training was done with a standalone build instead.

---

## Lessons learned

- **Exploration matters.** In the first version the brake action was active about half the time under random actions, so the car hardly moved, never got closer to the spot and the reward stayed flat at exactly −1.0. Making the brake activate only above a threshold fixed learning immediately.
- **Reading the reward numbers helps debugging.** A mean reward of exactly −1.000 with almost zero standard deviation meant every episode was timing out, because the step penalties sum to −1.
- **Parallel environments** gave a large speed-up: 10k steps took ~75 s with one area and ~8 s with nine.

---

## Repository structure

```
Assets/
  Scenes/SampleScene.unity   parking environment (9 training areas)
  Scripts/CarAgent.cs        agent: observations, actions, rewards
  Materials/                 colours for the car, spot and obstacles
  Models/CarParking.onnx     trained policy
config/
  car_parking.yaml           baseline PPO configuration
  car_parking_small.yaml     smaller network
  car_parking_lr.yaml        higher learning rate
```
