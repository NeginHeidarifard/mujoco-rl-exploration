"""
Perception Pipeline Demonstration (exploratory).

This script demonstrates a minimal visual-observation processing module
alongside the PPO control loop:

1. RGB visual observations are rendered from the MuJoCo environment.
2. A CNN encoder processes the visual observations into feature vectors.
3. The trained PPO policy (state-based) produces actions in closed-loop control.

The CNN encoder is NOT yet connected to the policy network. The current
PPO controller remains state-based. This script establishes the visual
observation processing infrastructure as a step toward end-to-end visual
policy learning, which is planned as future work.
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO


class VisualEncoder(nn.Module):
    """CNN encoder that maps RGB frames to feature vectors."""

    def __init__(self, output_dim=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, output_dim),
        )

    def forward(self, x):
        return self.cnn(x)


def preprocess_frame(frame):
    """Convert RGB frame (H, W, 3) to tensor (1, 3, H, W) normalized to [0, 1]."""
    frame_array = np.ascontiguousarray(frame.transpose(2, 0, 1))
    tensor = torch.from_numpy(frame_array).float()
    tensor = tensor.unsqueeze(0) / 255.0
    return tensor


def main():
    env_id = "InvertedPendulum-v4"
    model_path = "models/ppo_inverted_pendulum.zip"

    env = gym.make(env_id, render_mode="rgb_array")
    encoder = VisualEncoder(output_dim=64)
    encoder.eval()

    model = PPO.load(model_path)

    print("\n" + "=" * 60)
    print("PERCEPTION PIPELINE DEMONSTRATION (exploratory)")
    print("=" * 60 + "\n")

    n_episodes = 3

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=episode)
        done = False
        step = 0
        total_reward = 0.0
        visual_features = None

        while not done:
            # Step 1: Render visual observation (RGB frame)
            frame = env.render()

            # Step 2: Encode visual observation through CNN
            frame_tensor = preprocess_frame(frame)
            with torch.no_grad():
                visual_features = encoder(frame_tensor).numpy().flatten()

            # Step 3: Action selection (state-based PPO controller)
            action, _ = model.predict(obs, deterministic=True)

            # Step 4: Environment transition
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            step += 1

        print(
            f"Episode {episode + 1}: "
            f"steps={step}, reward={total_reward:.2f}, "
            f"visual_feature_dim={visual_features.shape[0]}"
        )

    env.close()

    print("\n" + "=" * 60)
    print("Visual observations were rendered and encoded by a CNN.")
    print("The PPO controller remains state-based in this exploratory version.")
    print("End-to-end visual policy learning is planned as the next step.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()