"""
Evaluate trained PPO policy on MuJoCo InvertedPendulum-v4.
"""

import gymnasium as gym
from stable_baselines3 import PPO


def main():
    env_id = "InvertedPendulum-v4"
    model_path = "models/ppo_inverted_pendulum.zip"

    env = gym.make(env_id, render_mode="human")

    model = PPO.load(model_path)

    n_episodes = 5

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated
            total_reward += reward

        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()