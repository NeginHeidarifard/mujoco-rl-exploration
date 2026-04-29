"""
Evaluate a lightweight visual-shift proxy for MuJoCo control.

Important scope:
- The trained PPO policy is state-based.
- The policy does NOT consume image observations directly.
- Rendered RGB frames are used only to estimate visual degradation severity.
- That severity is mapped to synthetic corruption of the numerical observation vector.

This is a minimal proxy experiment for studying how degraded visual perception
could affect downstream closed-loop control behavior.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from stable_baselines3 import PPO


@dataclass
class VisualCondition:
    name: str
    blur_radius: float = 0.0
    gaussian_noise_std: float = 0.0
    occlusion_ratio: float = 0.0
    brightness_scale: float = 1.0


def apply_visual_shift(
    frame: np.ndarray,
    condition: VisualCondition,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply a visual degradation to an RGB frame."""

    out = frame.astype(np.float32)

    if condition.blur_radius > 0:
        image = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
        image = image.filter(ImageFilter.GaussianBlur(radius=condition.blur_radius))
        out = np.asarray(image).astype(np.float32)

    if condition.gaussian_noise_std > 0:
        out += rng.normal(
            loc=0.0,
            scale=condition.gaussian_noise_std,
            size=out.shape,
        )

    if condition.occlusion_ratio > 0:
        height, width, _ = out.shape
        occ_h = max(1, int(height * np.sqrt(condition.occlusion_ratio)))
        occ_w = max(1, int(width * np.sqrt(condition.occlusion_ratio)))

        y0 = rng.integers(0, max(1, height - occ_h + 1))
        x0 = rng.integers(0, max(1, width - occ_w + 1))

        out[y0 : y0 + occ_h, x0 : x0 + occ_w, :] = 0.0

    if condition.brightness_scale != 1.0:
        out *= condition.brightness_scale

    return np.clip(out, 0, 255).astype(np.uint8)


def frame_shift_severity(clean_frame: np.ndarray, shifted_frame: np.ndarray) -> float:
    """Compute normalized frame difference using mean squared error."""

    clean = clean_frame.astype(np.float32) / 255.0
    shifted = shifted_frame.astype(np.float32) / 255.0

    mse = np.mean((clean - shifted) ** 2)

    return float(np.clip(mse, 0.0, 1.0))


def evaluate_condition(
    model: PPO,
    env: gym.Env,
    condition: VisualCondition,
    n_episodes: int = 10,
    severity_to_noise_gain: float = 2.5,
    seed: int = 42,
) -> dict:
    """
    Evaluate policy under a visual-shift proxy.

    Visual degradation severity is converted into observation noise before
    action selection.
    """

    episode_rewards = []
    visual_severities = []
    effective_obs_noise_stds = []

    rng = np.random.default_rng(seed)

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        total_reward = 0.0

        while not done:
            clean_frame = env.render()
            shifted_frame = apply_visual_shift(clean_frame, condition, rng)

            severity = frame_shift_severity(clean_frame, shifted_frame)
            obs_noise_std = severity_to_noise_gain * severity

            shifted_obs = obs + rng.normal(
                loc=0.0,
                scale=obs_noise_std,
                size=obs.shape,
            )

            action, _ = model.predict(shifted_obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)

            total_reward += reward
            visual_severities.append(severity)
            effective_obs_noise_stds.append(obs_noise_std)

            done = terminated or truncated

        episode_rewards.append(total_reward)

    return {
        "setting": condition.name,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_visual_severity": float(np.mean(visual_severities)),
        "mean_effective_obs_noise_std": float(np.mean(effective_obs_noise_stds)),
    }


def main() -> None:
    os.makedirs("outputs", exist_ok=True)

    env_id = "InvertedPendulum-v4"
    model_path = "models/ppo_inverted_pendulum.zip"

    env = gym.make(env_id, render_mode="rgb_array")
    model = PPO.load(model_path)

    conditions = [
        VisualCondition(name="clean"),
        VisualCondition(name="blurred", blur_radius=8.0),
        VisualCondition(name="noisy", gaussian_noise_std=80.0),
        VisualCondition(name="occluded", occlusion_ratio=0.55),
        VisualCondition(name="dark", brightness_scale=0.25),
        VisualCondition(name="bright", brightness_scale=2.00),
    ]

    results = []

    for condition in conditions:
        result = evaluate_condition(
            model=model,
            env=env,
            condition=condition,
            n_episodes=10,
            severity_to_noise_gain=2.5,
        )

        results.append(result)

        print(
            f"{result['setting']:>8}: "
            f"reward = {result['mean_reward']:.2f} ± {result['std_reward']:.2f}, "
            f"visual severity = {result['mean_visual_severity']:.4f}, "
            f"effective obs noise = {result['mean_effective_obs_noise_std']:.4f}"
        )

    df = pd.DataFrame(results)

    csv_path = "outputs/visual_shift_evaluation_results.csv"
    plot_path = "outputs/visual_shift_evaluation.png"

    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(df["setting"], df["mean_reward"])
    plt.ylabel("Mean episode reward")
    plt.title("Visual Shift Proxy Evaluation")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    env.close()

    print(f"Saved results to {csv_path}")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()