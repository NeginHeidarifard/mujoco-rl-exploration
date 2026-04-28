# MuJoCo RL Exploration

Minimal reinforcement learning exploration using MuJoCo and Stable-Baselines3.

## Project Goal

This repository contains a small and reproducible reinforcement learning experiment in a standard MuJoCo continuous-control environment.

The goal is not to propose a new robotics method, but to build practical familiarity with:

- MuJoCo simulation environments
- PPO training with Stable-Baselines3
- closed-loop policy learning
- reward logging and evaluation
- basic policy behavior in controlled simulated dynamics

This project supports a broader research interest in perception-to-action systems and robust decision-making under distribution shift.

## Environment

The initial experiment uses `InvertedPendulum-v4`.

This is a standard MuJoCo continuous-control task where the agent learns to keep a pendulum upright by controlling the cart.

## Algorithm

The policy is trained using PPO, Proximal Policy Optimization.

Implementation details:

- Library: Stable-Baselines3
- Policy: MLP Policy
- Device: CPU
- Training timesteps: 50,000
- Evaluation episodes: 5

## Results

### Training Reward Curve

The reward curve shows clear learning progression and convergence toward stable control performance.

![Reward Curve](outputs/reward_curve.png)

### Evaluation

The trained policy was evaluated for 5 deterministic episodes.

- Episode 1: Total Reward = 1000.00
- Episode 2: Total Reward = 1000.00
- Episode 3: Total Reward = 1000.00
- Episode 4: Total Reward = 1000.00
- Episode 5: Total Reward = 1000.00

The final policy consistently reaches the maximum reward in this environment, indicating successful stabilization of the pendulum.

## Repository Structure

- `train_ppo.py`: PPO training script
- `evaluate.py`: deterministic policy evaluation script
- `NOTES.md`: learning notes and interview explanation
- `requirements.txt`: Python dependencies
- `models/ppo_inverted_pendulum.zip`: trained PPO model
- `outputs/reward_log.csv`: episode reward log
- `outputs/reward_curve.png`: reward curve plot

## How to Run

Create and activate the environment:

    conda create -n mujoco-rl python=3.11.5 -y
    conda activate mujoco-rl
    pip install -r requirements.txt

Train the PPO agent:

    python train_ppo.py

Evaluate the trained policy:

    python evaluate.py

## Scope and Limitations

This project is intentionally minimal.

It does not claim:

- a novel RL algorithm
- a full perception-to-action robotics pipeline
- visual-policy learning
- real-robot deployment
- state-of-the-art robotics performance

It is an initial simulation-based RL exploration designed to support future work on perception-to-action systems under distribution shift.
