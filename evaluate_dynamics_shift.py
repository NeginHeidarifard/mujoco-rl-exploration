"""
Evaluate a trained PPO policy under simple train-test mismatch settings.

This script modifies MuJoCo dynamics parameters and optionally adds
observation noise at evaluation time.

It is an exploratory robustness/sensitivity check, not a formal
distribution-shift benchmark.
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


class ObservationNoiseWrapper(gym.ObservationWrapper):
    """Add Gaussian noise to observations at inference time."""

    def __init__(self, env, noise_std=0.0):
        super().__init__(env)
        self.noise_std = noise_std

    def observation(self, obs):
        if self.noise_std <= 0:
            return obs
        noise = np.random.normal(0, self.noise_std, size=obs.shape)
        return obs + noise


def make_shifted_env(mass_scale=1.0, gravity_scale=1.0, noise_std=0.0):
    """Create an InvertedPendulum environment with simple dynamics shifts."""
    env = gym.make("InvertedPendulum-v4")

    # Dynamics shift: scale body masses
    env.unwrapped.model.body_mass[:] *= mass_scale

    # Dynamics shift: scale gravity vector
    env.unwrapped.model.opt.gravity[:] *= gravity_scale

    # Observation perturbation
    if noise_std > 0:
        env = ObservationNoiseWrapper(env, noise_std=noise_std)

    return env


def run_eval(env, model, episodes=10):
    """Evaluate a fixed policy and return mean/std episode reward."""
    rewards = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        rewards.append(total_reward)

    return float(np.mean(rewards)), float(np.std(rewards))


def main():
    model = PPO.load("models/ppo_inverted_pendulum.zip")

    configs = [
        {"name": "clean", "mass": 1.0, "gravity": 1.0, "noise": 0.0},
        {"name": "mass_low", "mass": 0.75, "gravity": 1.0, "noise": 0.0},
        {"name": "mass_high", "mass": 2.0, "gravity": 1.0, "noise": 0.0},
        {"name": "gravity_high", "mass": 1.0, "gravity": 1.5, "noise": 0.0},
        {"name": "mass_gravity_high", "mass": 2.0, "gravity": 1.5, "noise": 0.0},
        {"name": "mass_gravity_noise", "mass": 2.0, "gravity": 1.5, "noise": 0.05},
        {"name": "extreme_shift", "mass": 3.0, "gravity": 2.0, "noise": 0.10},
    ]

    print("Dynamics and Observation Shift Evaluation")
    print("=========================================")

    for cfg in configs:
        env = make_shifted_env(
            mass_scale=cfg["mass"],
            gravity_scale=cfg["gravity"],
            noise_std=cfg["noise"],
        )

        mean, std = run_eval(env, model, episodes=10)
        env.close()

        print(
            f"{cfg['name']:20s} | "
            f"mass={cfg['mass']:.2f} | "
            f"gravity={cfg['gravity']:.2f} | "
            f"noise={cfg['noise']:.2f} | "
            f"mean_reward={mean:.2f} | "
            f"std={std:.2f}"
        )


if __name__ == "__main__":
    main()