# MuJoCo RL Exploration

A minimal, reproducible reinforcement learning exploration in MuJoCo using Stable-Baselines3.

This repository contains:
- a state-based PPO baseline,
- sensitivity analysis under observation perturbations,
- an exploratory visual PPO experiment trained directly on rendered RGB frames.

> Limitations and scope are listed explicitly in [`LIMITATIONS.md`](LIMITATIONS.md).

---

## Project Goal

This project aims to build practical familiarity with:

- MuJoCo continuous-control environments
- PPO training with Stable-Baselines3
- closed-loop policy behavior
- sensitivity of learned policies to observation quality
- basic perception-to-action pipelines from pixels to actions

The goal is not to propose a new method, but to construct a clean, reproducible experimental setup for studying how observation quality affects control performance.

---

## Environment

Experiments are conducted on `InvertedPendulum-v4`, a standard MuJoCo continuous-control task where the agent learns to keep a pendulum upright.

Two observation modalities are used:

- **State observations** (default low-dimensional vector)
- **Pixel observations** (rendered RGB frames)

---

## Methods

### 1. State-Based PPO Baseline

- Algorithm: PPO (Stable-Baselines3)
- Policy: MLP
- Timesteps: 50,000
- Device: CPU

**Final evaluation:** 1000.00 reward across all 5 episodes.

![Reward Curve](outputs/reward_curve.png)

This confirms the task is solved under state observations.

---

### 2. Observation Noise Sensitivity

The trained state-based policy is evaluated under Gaussian noise added to observations at inference time.

| Noise sigma | Mean Reward | Std |
|--------|------------|-----|
| 0.00   | 1000.00    | 0.00 |
| 0.05   | 1000.00    | 0.00 |
| 0.10   | 81.90      | 81.33 |
| 0.20   | 13.90      | 5.87 |
| 0.30   | 8.90       | 5.05 |

**Observation:** Performance collapses beyond a small noise threshold (around 0.05-0.10), indicating strong sensitivity to observation quality. This is a single empirical observation, not a general robustness claim.

---

### 3. Observation Perturbation Experiments

Additional evaluation of the state-based policy under:

- Gaussian noise
- Partial observation masking

Small perturbations have minimal effect; missing state information causes strong degradation.

![Shift Evaluation](outputs/shift_evaluation.png)

---

### 4. Visual Degradation Proxy

A lightweight experiment that:

1. renders RGB frames from the environment,
2. applies visual degradations (blur, noise, occlusion, brightness),
3. measures degradation severity,
4. maps that severity to noise on the **state vector** before policy inference.

**Note:** this script evaluates the **state-based** policy. It does not train or evaluate an image-based policy.

![Visual Shift Evaluation](outputs/visual_shift_evaluation.png)

---

### 5. Exploratory Visual PPO (Pixel-Based Policy)

A PPO agent is trained directly on RGB frames:

- Policy: CnnPolicy
- Input: 64x64 RGB images
- Timesteps: 20,000 (exploratory short run)
- Device: CPU

![Visual PPO Reward Curve](outputs/visual_reward_curve.png)

**Interpretation:**

- reward increases from ~5 to ~20-25 (rolling mean), peaks ~85
- indicates partial learning from pixels
- **not fully converged** - a converged visual policy on this task typically requires 500K-1M+ timesteps

This experiment demonstrates the end-to-end pipeline:

RGB observations -> CNN policy -> action

---

### 6. Visual Encoder Demonstration

`perception_pipeline.py` demonstrates:

- RGB rendering
- CNN feature extraction
- closed-loop execution with the state-based policy

The CNN encoder is not integrated into the policy network in this script. It serves as a standalone demonstration of visual processing infrastructure.

---

## Repository Structure

**Scripts:**
- `train_ppo.py` - state-based PPO training
- `train_visual_ppo.py` - exploratory visual PPO (CnnPolicy)
- `pixel_wrapper.py` - RGB observation wrapper
- `wrappers.py` - Gaussian observation noise wrapper
- `evaluate.py` - deterministic evaluation
- `evaluate_observation_noise.py` - sensitivity analysis under noise
- `evaluate_shift.py` - observation perturbation evaluation
- `evaluate_visual_shift.py` - visual degradation proxy (state-based)
- `perception_pipeline.py` - CNN encoder demonstration

**Models:**
- `models/ppo_inverted_pendulum.zip` - state-based PPO
- `models/ppo_visual_inverted_pendulum.zip` - exploratory visual PPO

**Outputs:**
- `outputs/reward_curve.png` - state-based training curve
- `outputs/visual_reward_curve.png` - visual training curve
- `outputs/*.csv` - reward logs and evaluation results
- `outputs/*.png` - additional evaluation plots

---

## How to Run

Setup the environment:

    conda create -n mujoco-rl python=3.11.5 -y
    conda activate mujoco-rl
    pip install -r requirements.txt

Train baselines:

    python train_ppo.py
    python train_visual_ppo.py

Evaluate:

    python evaluate.py
    python evaluate_observation_noise.py
    python evaluate_shift.py
    python evaluate_visual_shift.py
    python perception_pipeline.py

---

## Scope

This is an exploratory project.

**It does not claim:**
- a novel RL algorithm
- fully converged visual control
- formal distribution-shift analysis
- adaptation, fine-tuning, or meta-learning
- sim-to-real transfer
- real robot deployment

**It provides:**
- a reproducible state-based PPO baseline
- empirical sensitivity analysis under observation noise
- a working but unconverged pixel-to-action training pipeline
- a standalone CNN encoder demonstration
- explicit limitations (see [`LIMITATIONS.md`](LIMITATIONS.md))

---

## Future Work

- longer visual PPO training (>=500K timesteps)
- integration of the CNN encoder directly into the policy network
- rigorous comparison of state vs. pixel robustness
- evaluation under formal distribution shifts (domain randomization, sim-to-real)
- multi-seed evaluation and statistical analysis
- scaling to more complex MuJoCo tasks (HalfCheetah, Hopper, Walker2D)
