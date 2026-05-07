"""
Compare standard PPO, domain-randomized PPO, and fine-tuned PPO under
controlled train-test dynamics and observation mismatch.
"""

import os
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

    original_mass = env.unwrapped.model.body_mass.copy()
    original_gravity = env.unwrapped.model.opt.gravity.copy()

    env.unwrapped.model.body_mass[:] = original_mass * mass_scale
    env.unwrapped.model.opt.gravity[:] = original_gravity * gravity_scale

    if noise_std > 0:
        env = ObservationNoiseWrapper(env, noise_std=noise_std)

    return env


def run_eval(env, model, episodes=10, seed=42):
    """Evaluate a fixed policy and return mean/std episode reward."""
    rewards = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
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
    os.makedirs("outputs", exist_ok=True)

    standard_model = PPO.load("models/ppo_inverted_pendulum.zip")
    randomized_model = PPO.load("models/ppo_domain_randomized.zip")
    finetuned_model = PPO.load("models/ppo_finetuned_shift.zip")

    models = {
        "standard": standard_model,
        "randomized": randomized_model,
        "finetuned": finetuned_model,
    }

    configs = [
        {"name": "clean", "mass": 1.0, "gravity": 1.0, "noise": 0.0},
        {"name": "gravity_shift", "mass": 1.0, "gravity": 1.5, "noise": 0.0},
        {"name": "combined_shift", "mass": 2.0, "gravity": 1.5, "noise": 0.0},
        {"name": "combined_noise", "mass": 2.0, "gravity": 1.5, "noise": 0.05},
        {"name": "extreme_shift", "mass": 3.0, "gravity": 2.0, "noise": 0.10},
    ]

    results = []

    print("\nRobustness under Train-Test Dynamics Mismatch")
    print("=" * 60)

    for policy_name, model in models.items():
        for cfg in configs:
            env = make_shifted_env(
                mass_scale=cfg["mass"],
                gravity_scale=cfg["gravity"],
                noise_std=cfg["noise"],
            )

            mean_reward, std_reward = run_eval(env, model, episodes=10)
            env.close()

            results.append(
                {
                    "setting": cfg["name"],
                    "policy": policy_name,
                    "mass_scale": cfg["mass"],
                    "gravity_scale": cfg["gravity"],
                    "noise_std": cfg["noise"],
                    "mean_reward": mean_reward,
                    "std_reward": std_reward,
                }
            )

            print(
                f"{policy_name:12s} | "
                f"{cfg['name']:16s} | "
                f"mean={mean_reward:8.2f} | "
                f"std={std_reward:7.2f}"
            )

    df = pd.DataFrame(results)

    csv_path = "outputs/domain_randomization_results.csv"
    plot_path = "outputs/domain_randomization_comparison.png"

    df.to_csv(csv_path, index=False)

    pivot_df = df.pivot(
        index="setting",
        columns="policy",
        values="mean_reward",
    )

    ordered_settings = [
        "clean",
        "gravity_shift",
        "combined_shift",
        "combined_noise",
        "extreme_shift",
    ]

    ordered_policies = [
        "standard",
        "randomized",
        "finetuned",
    ]

    pivot_df = pivot_df.loc[ordered_settings, ordered_policies]

    ax = pivot_df.plot(kind="bar", figsize=(10, 5))

    ax.set_ylabel("Mean episode reward")
    ax.set_xlabel("Evaluation setting")
    ax.set_title("Robustness under Train-Test Dynamics Mismatch")

    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print("=" * 60)
    print(f"Saved results to {csv_path}")
    print(f"Saved plot to {plot_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()