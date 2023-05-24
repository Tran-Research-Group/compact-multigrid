from abc import ABC, abstractmethod
from typing import Any, Final, NamedTuple, TypeVar, TypedDict, Union

import numpy as np
from numpy.typing import NDArray
import jax.numpy as jnp
from jax.typing import ArrayLike
import gymnasium as gym
from gymnasium import spaces
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from compact_multigrid.typing import Metadata, RenderMode, Observation, Direction

ObjectId = TypeVar("ObjectId", bound=Union[dict[str, int], TypedDict])
Info = TypeVar("Info", bound=dict[str, Any])
Field = TypeVar("Field", bound=NamedTuple)


class BaseMultigrid(gym.Env, ABC):
    """
    Base Gymnasium environment for all multigrid environments.
    """

    metadata: Metadata = {
        "render_modes": ["human", "rgb_array", "ansi", "ascii"],
    }

    def __init__(
        self,
        map_path: str | None,
        num_max_steps: int,
        object_id: ObjectId,
        render_mode: RenderMode = None,
    ) -> None:
        """
        Initialize a new multigrid environment.

        Parameters
        ----------
        num_max_steps : int
            The maximum number of steps per episode.
        render_mode : RenderMode = None
            The render mode to use

        Attributes
        ----------
        action_space : spaces.Discrete
            The discrete action space of the environment
        """
        super().__init__()

        self._object_id: Final[ObjectId] = object_id
        self._num_max_steps: Final[int] = num_max_steps
        self._map_path: Final[str | None] = map_path

        self._field_map: ArrayLike = self._load_field_map(map_path)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self._render_mode: Final[RenderMode] = render_mode

        self.observation_space: Final[spaces.Space] = self._define_observation_space()

        actions: Final[tuple[Direction, ...]] = self._define_actions()
        self.action_space: Final = spaces.Discrete(len(actions))

    @abstractmethod
    def _load_field_map(self, map_path: str | None) -> NDArray[np.integer]:
        """
        Abstract method to load the field map.

        Returns
        -------
        field_map : ArrayLike
            the field map
        """
        ...

    @abstractmethod
    def _define_observation_space(self) -> spaces.Space:
        """
        Abstract method to define the observation space of the environment.

        Returns
        -------
        action_space : spaces.Space
        """
        ...

    @abstractmethod
    def _define_actions(self) -> tuple[Direction, ...]:
        """
        Abstract method to define the actions of the environment.

        Returns
        -------
        actions : list[Direction]
            a list of actions to certain directions
        """
        ...

    @abstractmethod
    def _get_obs(self) -> Observation:
        """
        Abstract method to get the current observation of the environment.

        Returns
        -------
        observation : Observation
            the observation of the current state of the environment
        """
        ...

    @abstractmethod
    def _get_info(self) -> Info:
        """
        Abstract method to get the current information of the environment.

        Returns
        -------
        info : Info
            the information of the current state of the environment
        """
        ...

    @abstractmethod
    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, Info]:
        """
        Abstract method to reset the environment.

        Parameters
        ----------
        seed : int | None = None
            the seed to use for the random number generator
        options : dict[str, Any] | None = None
            additional options

        Returns
        -------
        observation : Observation
            the observation after the reset
        info : Info
            additional information
        """
        super().reset(seed=seed, options=options)

        self._step_count: int = 0
        self._episodic_reward: float = 0.0

        self._reset_field()

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    @abstractmethod
    def _reset_field(self) -> None:
        """
        Reset the game field to its initial state.
        """
        ...

    @abstractmethod
    def step(
        self, action: ArrayLike | int
    ) -> tuple[Observation, float, bool, bool, Info]:
        """
        Abstract method to execute a step in the environment.

        Parameters
        ----------
        action : ArrayLike
            the action to execute

        Returns
        -------
        observation : Observation
            the observation after the step
        reward : float
            the reward after the step
        terminated: bool
            whether the episode is terminated
        truncated: bool
            whether the episode is truncated
        info : Info
            additional information
        """
        self._step_count += 1
        ...

    @abstractmethod
    def render(self) -> ArrayLike | list[ArrayLike] | None:
        """
        Abstract method to render the environment.

        Returns
        -------
        image : ArrayLike | list[ArrayLike] | None
            the rendered image. If the render mode is "human", the plot is shown in GUI and image is None. If the render mode is "rgb_array", the image is returned as a numpy array. The render modes is "ansi" or "ascii" are not implemented in the base env.
        """
        fig, ax = self._render_grid()

        field: Field = self._get_field()

        fig.canvas.draw()

        image: ArrayLike | list[ArrayLike] | None = self._get_image(fig)

        return image

    @abstractmethod
    def _define_field(self) -> Field:
        """
        Abstract method to get the current game field of the environment.

        Returns
        -------
        field : Field
            the current game field of the environment
        """
        ...

    def _get_image(self, fig: Figure) -> ArrayLike | list[ArrayLike] | None:
        """
        Get the image of the rendered environment.

        Parameters
        ----------
        fig : Figure
            the figure of the plot

        Returns
        -------
        image: ArrayLike | list[ArrayLike] | None
            the rendered image. If the render mode is "human", the plot is shown in GUI and image is None. If the render mode is "rgb_array", the image is returned as a numpy array. The render modes is "ansi" or "ascii" are not implemented in the base env.
        """
        image: ArrayLike | list[ArrayLike] | None

        match self._render_mode:
            case "human":
                plt.show(block=False)
                image = None
            case "rgb_array":
                fig.canvas.draw()
                image = jnp.array(fig.canvas.renderer.buffer_rgba())  # type: ignore
            case _:
                image = None

        return image

    def _render_grid(self) -> tuple[Figure, Axes]:
        """
        Render the grid of the environment.

        Returns
        -------
        fig : Figure
            the figure of the plot
        ax : Axes
            the axes of the plot
        """
        h, w = self._get_map_shape()

        fig: Figure
        ax: Axes
        fig, ax = plt.subplots()

        ax.set_xlim(-0.5, w - 1)
        ax.set_ylim(-0.5, h - 1)
        ax.set_aspect(1)
        ax.invert_yaxis()
        ax.set_xticks(jnp.arange(0.5, w + 0.5, 1), minor=True)
        ax.set_yticks(jnp.arange(0.5, h + 0.5, 1), minor=True)
        ax.set_xticks(jnp.arange(0, w, 1))
        ax.set_yticks(jnp.arange(0, h, 1))
        ax.grid(which="minor")
        ax.tick_params(which="minor", length=0)

        return fig, ax

    @abstractmethod
    def _get_map_shape(self) -> tuple[int, int]:
        """
        Abstract method to get the shape of the map.

        Returns
        -------
        map_shape : tuple[int, int]
            the shape of the map (rwos, cols)
        """
        ...
