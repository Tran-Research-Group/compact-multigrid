from abc import ABC, abstractmethod

import gymnasium as gym
from gymnasium import spaces

from compact_multigrid.typing import Metadata, RenderMode

class BaseMultigrid(gym.Env, ABC):
    """
    Base Gymnasium environment for all multigrid environments.
    """

    def __init__(self) -> None:
        super().__init__()