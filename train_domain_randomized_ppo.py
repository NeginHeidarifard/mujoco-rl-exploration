"""
Train PPO with lightweight domain randomization on MuJoCo InvertedPendulum-v4.

Mass and gravity are randomized at each reset during training.
This is a simple train-time dynamics randomization baseline, not a formal
sim-to-real method.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback


class DomainRandomizationWrapper(gym.Wrapper):
    """Randomize mass and gravity at every environment reset."""

    def __init__(self, env):
        super().__init__(env)
        self.original_mass = env.unwrapped.model.body_mass.copy()
        self.original_gravity = env.unwrapped.model.opt.gravity.copy()

    def reset(self, **kwargs):
        mass_scale = np.random.uniform(0.7, 2.0)
        gravity_scale = np.random.uniform(0.8, 1.6)

        self.env.unwrapped.model.body_mass[:] = self.original_mass * mass_scale
        self.env.unwrapped.model.opt.gravity[:] = self.original_gravity * gravity_scale

        return self.env.reset(**kwargs)


class RewardLoggerCallback(BaseCallback):
    """Callback for logging episode rewards during training."""

    def __init__(self, log_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = log_path
        self.episode_rewards = []
        self.timesteps = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.timesteps.append(self.num_timesteps)

        return True

    def _on_training_end(self) -> None:
        if len(self.episode_rewards) == 0:
            print("No completed episodes were logged.")
            return

        df = pd.DataFrame(
            {
                "timesteps": self.timesteps,
                "episode_reward": self.episode_rewards,
            }
        )
        df.to_csv(self.log_path, index=False)
        print(f"Saved reward log to {self.log_path}")


def plot_reward_curve(csv_path: str, output_path: str) -> None:
    """Plot episode reward curve from CSV log."""

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(8, 5))
    plt.plot(df["timesteps"], df["episode_reward"], label="Episode reward")

    if len(df) >= 5:
        rolling = df["episode_reward"].rolling(window=5).mean()
        plt.plot(df["timesteps"], rolling, label="Rolling mean (window=5)")

    plt.xlabel("Timesteps")
    plt.ylabel("Episode reward")
    plt.title("Domain-Randomized PPO on InvertedPendulum-v4")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved reward curve to {output_path}")


def main() -> None:
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    env_id = "InvertedPendulum-v4"
    total_timesteps = 50_000
    seed = 42

    np.random.seed(seed)

    env = gym.make(env_id)
    env = DomainRandomizationWrapper(env)
    env = Monitor(env)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=seed,
        device="cpu",
        n_steps=1024,
        batch_size=64,
        learning_rate=3e-4,
    )

    reward_log_path = "outputs/domain_randomized_reward_log.csv"
    reward_curve_path = "outputs/domain_randomized_reward_curve.png"
    model_path = "models/ppo_domain_randomized"

    callback = RewardLoggerCallback(log_path=reward_log_path)

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=False,
    )

    model.save(model_path)
    print(f"Saved trained model to {model_path}.zip")

    env.close()

    plot_reward_curve(
        csv_path=reward_log_path,
        output_path=reward_curve_path,
    )


if __name__ == "__main__":
    main()