# MuJoCo RL Exploration

Minimal reinforcement learning exploration using MuJoCo and Stable-Baselines3.

## Project Goal

This repository contains a small and reproducible reinforcement learning experiment in a standard MuJoCo continuous-control environment.

The goal is not to propose a new robotics method, but to build practical familiarity with:

- MuJoCo simulation environments
- PPO training with Stable-Baselines3
- closed-loop policy learning
- reward logging and evaluation
- policy behavior under observation perturbations

This project supports a broader research interest in perception-to-action systems and robust decision-making under distribution shift.

## Environment

The initial experiments use `InvertedPendulum-v4`.

This is a standard MuJoCo continuous-control task where the agent learns to keep a pendulum upright by controlling the cart.

## Method

### PPO Training

The policy is trained using PPO (Proximal Policy Optimization).

Implementation details:

- Library: Stable-Baselines3
- Policy: MLP Policy
- Device: CPU
- Training timesteps: 50,000
- Evaluation episodes: 5 deterministic runs

### Observation Shift Evaluation

A second experiment evaluates policy robustness under simple observation perturbations.

Three settings were compared:

- small Gaussian noise added to observations
- stronger Gaussian noise added to observations
- partial observation masking (one state variable removed)

This serves as a minimal proxy for studying how perception shifts affect downstream control behavior.

## Results

### Training Reward Curve

The reward curve shows clear learning progression and convergence toward stable control performance.

![Reward Curve](outputs/reward_curve.png)

### Deterministic Policy Evaluation

Final evaluation after training:

- Episode 1: Total Reward = 1000.00
- Episode 2: Total Reward = 1000.00
- Episode 3: Total Reward = 1000.00
- Episode 4: Total Reward = 1000.00
- Episode 5: Total Reward = 1000.00

The trained policy consistently reaches maximum reward, indicating stable control of the pendulum.

### Observation Shift Results

Policy performance under observation perturbation:

- Clean observations: 1000.00 ± 0.00
- Gaussian noise (0.01): 1000.00 ± 0.00
- Gaussian noise (0.05): 1000.00 ± 0.00
- Partial observation masking: 31.90 ± 5.15

Small Gaussian noise did not significantly affect performance, while removing one observation variable caused a strong performance drop.

This suggests that the learned policy is tolerant to small perturbations but highly sensitive to missing critical state information.

![Shift Evaluation](outputs/shift_evaluation.png)

## Repository Structure

- `train_ppo.py` — PPO training script
- `evaluate.py` — deterministic policy evaluation
- `evaluate_shift.py` — observation perturbation evaluation
- `NOTES.md` — technical notes and interview explanations
- `requirements.txt` — Python dependencies
- `models/ppo_inverted_pendulum.zip` — trained PPO model
- `outputs/reward_log.csv` — reward log
- `outputs/reward_curve.png` — training reward plot
- `outputs/shift_evaluation_results.csv` — robustness experiment results
- `outputs/shift_evaluation.png` — robustness comparison plot

## How to Run

Create and activate the environment:

    conda create -n mujoco-rl python=3.11.5 -y
    conda activate mujoco-rl
    pip install -r requirements.txt

Train the PPO agent:

    python train_ppo.py

Evaluate the trained policy:

    python evaluate.py

Run observation shift evaluation:

    python evaluate_shift.py

## Scope and Limitations

This project is intentionally minimal.

It does not claim:

- a novel RL algorithm
- a full visual perception-to-action robotics pipeline
- sim-to-real transfer
- real robot deployment
- state-of-the-art robotics performance

It is an initial simulation-based exploration designed to better understand how observation quality influences control behavior in closed-loop systems.
