from typing import Literal, TypeAlias

from gymnasium import spaces
import jax.numpy as jnp
from jax.numpy import integer

from compact_multigrid.typing import RenderMode
from compact_multigrid.envs.ctf.base import BaseCtf

EnemyPolicyMode: TypeAlias = Literal["fight", "patrol", "capture", "random", "none"]


class Ctf1v1(BaseCtf):
    """
    Gymnasium capture the flag environment.

    Element IDs:
    - BLUE_BACKGROUND = 0
    - RED_BACKGROUND = 1
    - BLUE_UGV = 2
    - BLUE_UAV = 3
    - RED_UGV = 4
    - RED_UAV = 5
    - BLUE_FLAG = 6
    - RED_FLAG = 7
    - OBSTACLE = 8
    """

    def __init__(
        self,
        map_path: str,
        enemy_policy_mode: EnemyPolicyMode,
        randomness: float = 0.75,
        capture_reward: float = 1.0,
        battle_reward_alpha: float = 0.25,
        obstacle_penalty_beta: float | None = None,
        step_penalty_gamma: float = 0.0,
        num_max_steps: int = 500,
        render_mode: RenderMode = None,
    ) -> None:
        super().__init__(map_path, num_max_steps, render_mode=render_mode)

        self._enemy_policy_mode = enemy_policy_mode

    def _define_observation_space(self) -> spaces.Dict:
        observation = spaces.Dict(
            {
                "blue_agent": spaces.Box(
                    low=jnp.array([-1, -1]),
                    high=jnp.array(self._field_map.shape) - 1,
                    dtype=integer,
                ),
                "red_agent": spaces.Box(
                    low=jnp.array([-1, -1]),
                    high=jnp.array(self._field_map.shape) - 1,
                    dtype=integer,
                ),
                "blue_flag": spaces.Box(
                    low=jnp.array(
                        [[0, 0] for _ in range(len(self.field.blue_flag))]
                    ).flatten(),
                    high=jnp.array(
                        [
                            self._field_map.shape
                            for _ in range(len(self.field.blue_flag))
                        ]
                    ).flatten()
                    - 1,
                    dtype=integer,
                ),
                "red_flag": spaces.Box(
                    low=jnp.array(
                        [[0, 0] for _ in range(len(self.field.red_flag))]
                    ).flatten(),
                    high=jnp.array(
                        [self._field_map.shape for _ in range(len(self.field.red_flag))]
                    ).flatten()
                    - 1,
                    dtype=integer,
                ),
                "blue_background": spaces.Box(
                    low=jnp.array(
                        [[0, 0] for _ in range(len(self.field.blue_background))]
                    ).flatten(),
                    high=jnp.array(
                        [
                            self._field_map.shape
                            for _ in range(len(self.field.blue_background))
                        ]
                    ).flatten()
                    - 1,
                    dtype=integer,
                ),
                "red_background": spaces.Box(
                    low=jnp.array(
                        [[0, 0] for _ in range(len(self.field.red_background))]
                    ).flatten(),
                    high=jnp.array(
                        [
                            self._field_map.shape
                            for _ in range(len(self.field.red_background))
                        ]
                    ).flatten()
                    - 1,
                    dtype=integer,
                ),
                "obstacle": spaces.Box(
                    low=jnp.array(
                        [[0, 0] for _ in range(len(self.field.obstacle))]
                    ).flatten(),
                    high=jnp.array(
                        [self._field_map.shape for _ in range(len(self.field.obstacle))]
                    ).flatten()
                    - 1,
                    dtype=integer,
                ),
                "is_red_agent_defeated": spaces.Discrete(2),
            }
        )

        return observation
