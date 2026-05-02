import gymnasium as gym
import numpy as np


class ObservationNoiseWrapper(gym.ObservationWrapper):
    """Add Gaussian noise to observations."""

    def __init__(self, env, noise_std=0.1):
        super().__init__(env)
        self.noise_std = noise_std

    def observation(self, obs):
        noise = np.random.normal(0, self.noise_std, size=obs.shape)
        return obs + noise
