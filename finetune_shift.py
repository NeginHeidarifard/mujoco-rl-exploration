"""
Fine-tune a pretrained PPO policy on a heavily shifted MuJoCo environment.

This is a minimal recovery baseline: a policy trained on the default
environment is fine-tuned on a modified environment (heavily scaled
mass and gravity).

The shift setting (mass=3.0, gravity=2.0) is chosen because it is the
setting where the pretrained policy clearly fails on, so any
improvement from fine-tuning is observable.

This is NOT meta-learning, online adaptation, or any sophisticated
adaptation mechanism. It is a simple supervised continuation of
training under a different dynamics setting.
"""

import os
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback


# Heavy shift setting where the pretrained policy actually fails
MASS_SCALE = 3.0
GRAVITY_SCALE = 2.0


class RewardLoggerCallback(BaseCallback):
    """Log episode rewards during fine-tuning."""

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
        print(f"Saved fine-tuning reward log to {self.log_path}")


def make_shifted_env(mass_scale=MASS_SCALE, gravity_scale=GRAVITY_SCALE):
    """Create a heavily shifted InvertedPendulum environment."""
    env = gym.make("InvertedPendulum-v4")
    env.unwrapped.model.body_mass[:] *= mass_scale
    env.unwrapped.model.opt.gravity[:] *= gravity_scale
    env = Monitor(env)
    return env


def evaluate_policy(env, model, episodes=10):
    """Run deterministic evaluation."""
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


def plot_finetune_curve(csv_path, output_path):
    """Plot fine-tuning reward curve."""
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(8, 5))
    plt.plot(df["timesteps"], df["episode_reward"], label="Episode reward", alpha=0.5)
    if len(df) >= 5:
        rolling = df["episode_reward"].rolling(window=5).mean()
        plt.plot(df["timesteps"], rolling, label="Rolling mean (window=5)", linewidth=2)
    plt.xlabel("Fine-tuning timesteps")
    plt.ylabel("Episode reward")
    plt.title(f"Fine-tuning PPO under shifted dynamics (mass x{MASS_SCALE}, gravity x{GRAVITY_SCALE})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved fine-tuning curve to {output_path}")


def main():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    pretrained_path = "models/ppo_inverted_pendulum.zip"
    finetuned_path = "models/ppo_finetuned_shift"
    log_path = "outputs/finetune_reward_log.csv"
    curve_path = "outputs/finetune_reward_curve.png"

    print("\n" + "=" * 60)
    print("FINE-TUNING UNDER DYNAMICS SHIFT")
    print(f"  Shift setting: mass x{MASS_SCALE}, gravity x{GRAVITY_SCALE}")
    print("=" * 60 + "\n")

    # Step 1: evaluate pretrained policy on shifted env
    print("Step 1: evaluating pretrained policy on shifted environment...")
    eval_env = make_shifted_env()
    model = PPO.load(pretrained_path, env=eval_env)
    pre_mean, pre_std = evaluate_policy(eval_env, model, episodes=10)
    eval_env.close()
    print(f"  Pretrained on shifted env:  reward = {pre_mean:.2f} +/- {pre_std:.2f}")

    # Step 2: fine-tune the policy on the shifted environment
    print("\nStep 2: fine-tuning the policy on the shifted environment...")
    train_env = make_shifted_env()
    model = PPO.load(pretrained_path, env=train_env)
    callback = RewardLoggerCallback(log_path=log_path)
    model.learn(total_timesteps=10_000, callback=callback, progress_bar=False)
    model.save(finetuned_path)
    train_env.close()
    print(f"  Saved fine-tuned model to {finetuned_path}.zip")

    # Step 3: evaluate the fine-tuned policy
    print("\nStep 3: evaluating fine-tuned policy on shifted environment...")
    eval_env2 = make_shifted_env()
    model = PPO.load(finetuned_path, env=eval_env2)
    post_mean, post_std = evaluate_policy(eval_env2, model, episodes=10)
    eval_env2.close()
    print(f"  Fine-tuned on shifted env:  reward = {post_mean:.2f} +/- {post_std:.2f}")

    # Step 4: summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Before fine-tuning: {pre_mean:7.2f} +/- {pre_std:6.2f}")
    print(f"After  fine-tuning: {post_mean:7.2f} +/- {post_std:6.2f}")
    print(f"Change:             {post_mean - pre_mean:+7.2f}")
    print("=" * 60 + "\n")
    print("Note: This is a simple fine-tuning baseline, not meta-learning")
    print("or online adaptation. See LIMITATIONS.md for full scope.")
    print("=" * 60 + "\n")

    # Plot the fine-tuning curve
    if os.path.exists(log_path):
        plot_finetune_curve(log_path, curve_path)


if __name__ == "__main__":
    main()
