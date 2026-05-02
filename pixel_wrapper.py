"""
Pixel observation wrapper for MuJoCo environments.

Replaces state observations with rendered RGB frames so that a CNN-based
policy can learn directly from pixels.
"""

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class PixelObservationWrapper(gym.ObservationWrapper):
    """Replace state observations with rendered RGB frames."""

    def __init__(self, env, image_size=64):
        super().__init__(env)
        self.image_size = image_size
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(image_size, image_size, 3),
            dtype=np.uint8,
        )

    def observation(self, obs):
        # Render RGB frame
        frame = self.env.render()

        # Resize to fixed size using PIL
        from PIL import Image as PILImage
        img = PILImage.fromarray(frame).resize(
            (self.image_size, self.image_size)
        )
        return np.array(img, dtype=np.uint8)
