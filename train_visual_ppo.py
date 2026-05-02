"""
Train a CNN-based PPO policy directly on rendered RGB frames.

This is an end-to-end visual policy learning experiment:
- Observations: RGB frames from MuJoCo (64x64x3)
- Policy: CnnPolicy (Stable-Baselines3 default CNN)
- Algorithm: PPO
- Device: CPU

This is an exploratory short run to demonstrate the visual learning
pipeline. Training a fully-converged visual policy requires significantly
more timesteps than used here.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from pixel_wrapper import PixelObservationWrapper


class RewardLoggerCallback(BaseCallback):
    """Log episode rewards during training."""

    def __init__(self, log_path, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.episode_rewards = []
        self.timesteps = []

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.timesteps.append(self.num_timesteps)
        return True

    def _on_training_end(self):
        if not self.episode_rewards:
            print("No completed episodes were logged.")
            return
        df = pd.DataFrame({
            "timesteps": self.timesteps,
            "episode_reward": self.episode_rewards,
        })
        df.to_csv(self.log_path, index=False)
        print(f"Saved reward log to {self.log_path}")


def make_pixel_env():
    """Create pixel-based InvertedPendulum environment."""
    env = gym.make("InvertedPendulum-v4", render_mode="rgb_array")
    env = PixelObservationWrapper(env, image_size=64)
    env = Monitor(env)
    return env


def plot_reward_curve(csv_path, output_path):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(8, 5))
    plt.plot(df["timesteps"], df["episode_reward"], label="Episode reward")
    if len(df) >= 5:
        rolling = df["episode_reward"].rolling(window=5).mean()
        plt.plot(df["timesteps"], rolling, label="Rolling mean (window=5)")
    plt.xlabel("Timesteps")
    plt.ylabel("Episode reward")
    plt.title("Visual PPO on InvertedPendulum-v4 (CnnPolicy)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved reward curve to {output_path}")


def main():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    total_timesteps = 20_000  # short exploratory run
    seed = 42

    env = make_pixel_env()

    print("\n" + "=" * 60)
    print("VISUAL PPO TRAINING (CnnPolicy on RGB frames)")
    print("=" * 60)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space:      {env.action_space}")
    print(f"Total timesteps:   {total_timesteps}")
    print("=" * 60 + "\n")

    model = PPO(
        policy="CnnPolicy",
        env=env,
        verbose=1,
        seed=seed,
        device="cpu",
        n_steps=512,
        batch_size=64,
        learning_rate=3e-4,
    )

    reward_log_path = "outputs/visual_reward_log.csv"
    reward_curve_path = "outputs/visual_reward_curve.png"
    model_path = "models/ppo_visual_inverted_pendulum"

    callback = RewardLoggerCallback(log_path=reward_log_path)

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=False,
    )

    model.save(model_path)
    print(f"\nSaved trained model to {model_path}.zip")

    env.close()

    plot_reward_curve(
        csv_path=reward_log_path,
        output_path=reward_curve_path,
    )


if __name__ == "__main__":
    main()
