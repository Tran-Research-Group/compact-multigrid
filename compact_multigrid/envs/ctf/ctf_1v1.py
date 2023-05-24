import random
from typing import Any, Final, Literal, TypeAlias
from typing_extensions import TypedDict

from gymnasium import spaces
import jax.numpy as jnp
from jax.numpy import integer
import numpy as np
from compact_multigrid.policy.probablistic import (
    CapturePolicy,
    FightPolicy,
    PatrolPolicy,
)

from compact_multigrid.typing import Observation, RenderMode
from compact_multigrid.envs.ctf.base_ctf import BaseCtf, Field
from compact_multigrid.typing.field import Location
from compact_multigrid.typing.info import Observation
from compact_multigrid.utils import (
    distance_area_area,
    distance_points,
    distance_area_point,
)

EnemyPolicyMode: TypeAlias = Literal["fight", "patrol", "capture", "random", "none"]
EnemyPolicy: TypeAlias = FightPolicy | PatrolPolicy | CapturePolicy
InfoDict: TypeAlias = dict[str, float]


class ResetOptions(TypedDict):
    blue_agent_loc: Location | None
    red_agent_loc: Location | None


default_reset_options: ResetOptions = {
    "blue_agent_loc": None,
    "red_agent_loc": None,
}


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
        is_move_clipped: bool = True,
        num_max_steps: int = 500,
        render_mode: RenderMode = None,
    ) -> None:
        super().__init__(map_path, num_max_steps, render_mode=render_mode)

        self._enemy_policy_mode: Final[EnemyPolicyMode] = enemy_policy_mode

        self._randomness: Final[float] = randomness
        self._capture_reward: Final[float] = capture_reward
        self._battle_reward_alpha: Final[float] = battle_reward_alpha
        self._obstacle_penalty_beta: Final[float | None] = obstacle_penalty_beta
        self._step_penalty_gamma: Final[float] = step_penalty_gamma

        self._is_move_clipped: Final[bool] = is_move_clipped

        self._enemy_policy: EnemyPolicy

        match enemy_policy_mode:
            case "capture":
                self._enemy_policy = CapturePolicy(
                    self.field, self._field_map, self._randomness
                )
            case "fight":
                self._enemy_policy = FightPolicy(
                    self.field, self._field_map, self._randomness
                )
            case "patrol":
                self._enemy_policy = PatrolPolicy(
                    self.field, self._field_map, self._randomness
                )
            case "none":
                pass
            case _:
                raise Exception(
                    f"[compact-multigrid] The enemy policy {enemy_policy_mode} is not defined."
                )

    def _define_observation_space(self) -> spaces.Dict:
        observation = spaces.Dict(
            {
                "blue_agent": spaces.Box(
                    low=np.array([-1, -1]),
                    high=np.array(self._field_map.shape) - 1,
                    dtype=integer,
                ),
                "red_agent": spaces.Box(
                    low=np.array([-1, -1]),
                    high=np.array(self._field_map.shape) - 1,
                    dtype=integer,
                ),
                "blue_flag": spaces.Box(
                    low=np.array(
                        [[0, 0] for _ in range(len(self.field.blue_flag))]
                    ).flatten(),
                    high=np.array(
                        [
                            self._field_map.shape
                            for _ in range(len(self.field.blue_flag))
                        ]
                    ).flatten()
                    - 1,
                    dtype=integer,
                ),
                "red_flag": spaces.Box(
                    low=np.array(
                        [[0, 0] for _ in range(len(self.field.red_flag))]
                    ).flatten(),
                    high=np.array(
                        [self._field_map.shape for _ in range(len(self.field.red_flag))]
                    ).flatten()
                    - 1,
                    dtype=integer,
                ),
                "blue_background": spaces.Box(
                    low=np.array(
                        [[0, 0] for _ in range(len(self.field.blue_background))]
                    ).flatten(),
                    high=np.array(
                        [
                            self._field_map.shape
                            for _ in range(len(self.field.blue_background))
                        ]
                    ).flatten()
                    - 1,
                    dtype=integer,
                ),
                "red_background": spaces.Box(
                    low=np.array(
                        [[0, 0] for _ in range(len(self.field.red_background))]
                    ).flatten(),
                    high=np.array(
                        [
                            self._field_map.shape
                            for _ in range(len(self.field.red_background))
                        ]
                    ).flatten()
                    - 1,
                    dtype=integer,
                ),
                "obstacle": spaces.Box(
                    low=np.array(
                        [[0, 0] for _ in range(len(self.field.obstacle))]
                    ).flatten(),
                    high=np.array(
                        [self._field_map.shape for _ in range(len(self.field.obstacle))]
                    ).flatten()
                    - 1,
                    dtype=integer,
                ),
                "is_red_agent_defeated": spaces.Discrete(2),
            }
        )

        return observation

    def _get_obs(self) -> Observation:
        observation: Observation = {
            "blue_agent": jnp.array(self._blue_agent_loc),
            "red_agent": jnp.array(self._red_agent_loc),
            "blue_flag": jnp.array(self.field.blue_flag).flatten(),
            "red_flag": jnp.array(self.field.red_flag).flatten(),
            "blue_background": jnp.array(self.field.blue_background).flatten(),
            "red_background": jnp.array(self.field.red_background).flatten(),
            "obstacle": jnp.array(self.field.obstacle).flatten(),
            "is_red_agent_defeated": jnp.array(int(self._is_red_agent_defeated)),
        }
        return observation

    def _get_info(self) -> InfoDict:
        info = {
            "d_ba_ra": distance_points(self._blue_agent_loc, self._red_agent_loc),
            "d_ba_bf": distance_area_point(self._blue_agent_loc, self.field.blue_flag),
            "d_ba_rf": distance_area_point(self._blue_agent_loc, self.field.red_flag),
            "d_ra_bf": distance_area_point(self._red_agent_loc, self.field.blue_flag),
            "d_ra_rf": distance_area_point(self._red_agent_loc, self.field.red_flag),
            "d_bf_rf": distance_area_area(self.field.blue_flag, self.field.red_flag),
            "d_ba_bb": distance_area_point(
                self._blue_agent_loc, self.field.blue_background
            ),
            "d_ba_rb": distance_area_point(
                self._blue_agent_loc, self.field.red_background
            ),
            "d_ra_bb": distance_area_point(
                self._red_agent_loc, self.field.blue_background
            ),
            "d_ra_rb": distance_area_point(
                self._red_agent_loc, self.field.red_background
            ),
            "d_ba_ob": distance_area_point(self._blue_agent_loc, self.field.obstacle),
        }
        return info

    def _update_field(self) -> Field:
        field = Field(
            self.field.blue_background,
            self.field.red_background,
            self._blue_agent_loc,
            None,
            self._red_agent_loc,
            None,
            self.field.blue_flag,
            self.field.red_flag,
            self.field.obstacle,
        )

        return field

    def reset(
        self,
        seed: int | None = None,
        options: ResetOptions = default_reset_options,
    ) -> tuple[Observation, InfoDict]:
        """
        Reset the environment.

        Parameters
        ----------
        seed : int | None = None
            Random seed.
        options : ResetOptions = default_reset_options
            Reset options. See `ResetOptions` for details.

        Returns
        -------
        observation : Observation
            Observation.
        info : InfoDict
            Information.
        """
        super().reset(seed=seed)
        self._blue_agent_loc = (
            options["blue_agent_loc"]
            if options["blue_agent_loc"] is not None
            else random.choice(self.field.blue_background)
        )
        self._red_agent_loc = (
            options["red_agent_loc"]
            if options["red_agent_loc"] is not None
            else random.choice(self.field.red_background)
        )

        self.blue_traj = [self._blue_agent_loc]
        self.red_traj = [self._red_agent_loc]

        self._is_red_agent_defeated: bool = (
            True if self._enemy_policy_mode == "none" else False
        )

        match self._enemy_policy_mode:
            case "none":
                pass
            case _:
                self._enemy_policy.reset(self._red_agent_loc)

        self.field = self._update_field()

        observation: Observation = self._get_obs()
        info: InfoDict = self._get_info()

        self.obs_list: list[Observation] = [observation]

        return observation, info

    def step(self, action: int) -> tuple[Observation, float, bool, bool, InfoDict]:
        """
        Step the environment.

        Parameters
        ----------
        action : int
            Action.

        Returns
        -------
        observation : Observation
            Observation.
        reward : float
            Reward.
        terminated : bool
            Whether the episode is terminated.
        truncated : bool
            Whether the episode is truncated.
        info : InfoDict
            Information.
        """

        if self._is_red_agent_defeated:
            pass
        else:
            match self._enemy_policy:
                case CapturePolicy():
                    self._red_agent_loc = self._enemy_policy.act()
                case FightPolicy():
                    self._red_agent_loc = self._enemy_policy.act(self._blue_agent_loc)
                case PatrolPolicy():
                    self._red_agent_loc = self._enemy_policy.act()
                case _:
                    raise Exception("[tl_search] The enemy policy is not defined.")

        direction = self.actions[action]

        new_blue_agent_loc = Location(
            self._blue_agent_loc.row + direction.row,
            self._blue_agent_loc.col + direction.col,
        )
        match self._is_move_clipped:
            case True:
                num_row: int
                num_col: int
                num_row, num_col = self._field_map.shape
                new_blue_agent_loc = Location(
                    jnp.clip(new_blue_agent_loc.row, 0, num_row - 1),
                    jnp.clip(new_blue_agent_loc.col, 0, num_col - 1),
                )

                if new_blue_agent_loc in self.field.obstacle:
                    pass
                else:
                    self._blue_agent_loc = new_blue_agent_loc

            case False:
                self._blue_agent_loc = new_blue_agent_loc

        self.blue_traj.append(self._blue_agent_loc)
        self.red_traj.append(self._red_agent_loc)

        reward, terminated, truncated = self._reward()

        self._step_count += 1
        self._episodic_reward += reward

        self.field = self._update_field()

        observation: Observation = self._get_obs()
        info: InfoDict = self._get_info()

        self.obs_list.append(observation)

        return observation, reward, terminated, truncated, info

    def _reward(self) -> tuple[float, bool, bool]:
        """
        Compute the reward.

        Returns
        -------
        reward : float
            Reward.
        terminated : bool
            Whether the episode is terminated.
        truncated : bool
            Whether the episode is truncated.
        """

        reward: float = 0.0
        terminated: bool = False
        truncated: bool = self._step_count >= self._num_max_steps

        if self._blue_agent_loc == self.field.red_flag:
            reward += 1.0
            terminated = True
        else:
            pass

        if self._red_agent_loc == self.field.blue_flag:
            reward -= 1.0
            terminated = True
        else:
            pass

        if (
            distance_points(self._blue_agent_loc, self._red_agent_loc) <= 1
            and not self._is_red_agent_defeated
        ):
            blue_win: bool

            match self._blue_agent_loc in self.field.blue_background:
                case True:
                    blue_win = np.random.choice(
                        [True, False], p=[self._randomness, 1.0 - self._randomness]
                    )
                case False:
                    blue_win = np.random.choice(
                        [False, True], p=[self._randomness, 1.0 - self._randomness]
                    )

            if blue_win:
                reward += self._battle_reward_alpha
                self._is_red_agent_defeated = True
            else:
                reward -= self._battle_reward_alpha
                terminated = True

        if self._obstacle_penalty_beta is not None:
            if self._blue_agent_loc in self.field.obstacle:
                reward -= self._obstacle_penalty_beta
                terminated = True
            else:
                pass
        else:
            pass

        reward -= self._step_penalty_gamma * 1

        return reward, terminated, truncated
