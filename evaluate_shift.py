"""
Evaluate PPO policy robustness under simple observation perturbations.

This script compares the trained policy on:
1. clean observations
2. noisy observations
3. partially masked observations

This is a minimal proxy experiment for studying how observation/perception shifts
can affect closed-loop control behavior.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gymnasium as gym
from stable_baselines3 import PPO


def evaluate_policy(model, env, n_episodes=10, noise_std=0.0, mask_index=None, seed=42):
    episode_rewards = []

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        total_reward = 0.0

        while not done:
            shifted_obs = obs.copy()

            if noise_std > 0:
                shifted_obs = shifted_obs + np.random.normal(
                    loc=0.0,
                    scale=noise_std,
                    size=shifted_obs.shape,
                )

            if mask_index is not None:
                shifted_obs[mask_index] = 0.0

            action, _ = model.predict(shifted_obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated
            total_reward += reward

        episode_rewards.append(total_reward)

    return episode_rewards


def main():
    os.makedirs("outputs", exist_ok=True)

    env_id = "InvertedPendulum-v4"
    model_path = "models/ppo_inverted_pendulum.zip"

    env = gym.make(env_id)
    model = PPO.load(model_path)

    experiments = {
        "clean": {"noise_std": 0.0, "mask_index": None},
        "noise_0.01": {"noise_std": 0.01, "mask_index": None},
        "noise_0.05": {"noise_std": 0.05, "mask_index": None},
        "masked_obs_1": {"noise_std": 0.0, "mask_index": 1},
    }

    results = []

    for name, config in experiments.items():
        rewards = evaluate_policy(
            model=model,
            env=env,
            n_episodes=10,
            noise_std=config["noise_std"],
            mask_index=config["mask_index"],
        )

        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)

        results.append(
            {
                "setting": name,
                "mean_reward": mean_reward,
                "std_reward": std_reward,
            }
        )

        print(f"{name}: mean reward = {mean_reward:.2f} ± {std_reward:.2f}")

    df = pd.DataFrame(results)
    df.to_csv("outputs/shift_evaluation_results.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(df["setting"], df["mean_reward"])
    plt.ylabel("Mean episode reward")
    plt.title("Policy Evaluation under Observation Shift")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("outputs/shift_evaluation.png", dpi=150)
    plt.close()

    env.close()

    print("Saved results to outputs/shift_evaluation_results.csv")
    print("Saved plot to outputs/shift_evaluation.png")


if __name__ == "__main__":
    main()