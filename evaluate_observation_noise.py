"""
Evaluate how Gaussian observation noise affects PPO policy performance.
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from wrappers import ObservationNoiseWrapper


def run_eval(env, model, episodes=10):
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
    model_path = "models/ppo_inverted_pendulum.zip"
    model = PPO.load(model_path)

    noise_levels = [0.0, 0.05, 0.10, 0.20, 0.30]

    print("Observation Noise Evaluation")
    print("============================")

    for noise_std in noise_levels:
        env = gym.make("InvertedPendulum-v4")

        if noise_std > 0:
            env = ObservationNoiseWrapper(env, noise_std=noise_std)

        mean_reward, std_reward = run_eval(env, model, episodes=10)
        env.close()

        print(f"noise_std={noise_std:.2f} | mean_reward={mean_reward:.2f} | std_reward={std_reward:.2f}")

    print("\nEvaluation completed. Inspect the reported mean/std rewards.")


if __name__ == "__main__":
    main()
