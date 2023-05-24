from typing import Final, NamedTuple, TypeGuard, TypedDict

import jax.numpy as jnp
from jax.typing import ArrayLike
from matplotlib.patches import Rectangle
import numpy as np
from numpy.typing import NDArray

from compact_multigrid import BaseMultigrid
from compact_multigrid.typing import Direction, RenderMode
from compact_multigrid.typing.field import Location
from compact_multigrid.utils import tuples2locs


class ObjectId(TypedDict):
    blue_background: int
    red_background: int
    blue_ugv: int
    blue_uav: int
    red_ugv: int
    red_uav: int
    blue_flag: int
    red_flag: int
    obstacle: int


class Field(NamedTuple):
    blue_background: list[Location]
    red_background: list[Location]
    blue_ugv: Location | None
    blue_uav: Location | None
    red_ugv: Location | None
    red_uav: Location | None
    blue_flag: list[Location]
    red_flag: list[Location]
    obstacle: list[Location]


default_object_id: ObjectId = {
    "blue_background": 0,
    "red_background": 1,
    "blue_ugv": 2,
    "blue_uav": 3,
    "red_ugv": 4,
    "red_uav": 5,
    "blue_flag": 6,
    "red_flag": 7,
    "obstacle": 8,
}


class BaseCtf(BaseMultigrid):
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
        num_max_steps: int,
        object_id: ObjectId = default_object_id,
        render_mode: RenderMode = None,
    ) -> None:
        super().__init__(map_path, num_max_steps, object_id, render_mode)

        self._field_map: NDArray[np.integer] = self._load_field_map(map_path)

        obstacle: Final[list[Location]] = tuples2locs(
            list(zip(*np.where(self._field_map == self._object_id["obstacle"])))  # type: ignore
        )
        blue_flag: Final[list[Location]] = tuples2locs(
            list(zip(*np.where(self._field_map == self._object_id["blue_flag"])))  # type: ignore
        )

        red_flag: Final[list[Location]] = tuples2locs(
            list(zip(*np.where(self._field_map == self._object_id["red_flag"])))  # type: ignore
        )
        blue_background: Final[list[Location]] = tuples2locs(
            list(zip(*np.where(self._field_map == self._object_id["blue_background"])))  # type: ignore
        ) + [blue_flag]
        red_background: Final[list[Location]] = tuples2locs(
            list(zip(*np.where(self._field_map == self._object_id["red_background"])))  # type: ignore
        ) + [red_flag]

        self.field = Field(
            blue_background,
            red_background,
            None,
            None,
            None,
            None,
            blue_flag,
            red_flag,
            obstacle,
        )

    def _load_field_map(self, map_path: str) -> NDArray[np.integer]:
        field_map: NDArray
        match map_path:
            case str():
                field_map = np.loadtxt(map_path, dtype=np.integer)
            case _:
                raise ValueError("[compact-multigrid] Invalid map path.")

        return field_map

    def _define_actions(self) -> tuple[Direction, ...]:
        """
        Define the actions that can be taken by the agent.

        0: stay, 1: left, 2: down, 3: right, 4: up
        """
        actions = (
            Direction(0, 0),
            Direction(0, -1),
            Direction(-1, 0),
            Direction(0, 1),
            Direction(1, 0),
        )

        return actions

    def _get_map_shape(self) -> tuple[int, int]:
        return self._field_map.shape

    def render(
        self, flag_markersize=30, agent_markersize=25
    ) -> ArrayLike | list[ArrayLike] | None:
        fig, ax = self._render_grid()

        (
            blue_background,
            red_background,
            blue_ugv,
            blue_uav,
            red_ugv,
            red_uav,
            blue_flag,
            red_flag,
            obstacle,
        ) = self._get_field()

        for obs in obstacle:
            obs_rec = Rectangle((obs.col - 0.5, obs.row - 0.5), 1, 1, color="black")
            ax.add_patch(obs_rec)

        for bb in blue_background:
            bb_rec = Rectangle((bb.col - 0.5, bb.row - 0.5), 1, 1, color="aliceblue")
            ax.add_patch(bb_rec)

        for rb in red_background:
            rf_rec = Rectangle((rb.col - 0.5, rb.row - 0.5), 1, 1, color="mistyrose")
            ax.add_patch(rf_rec)

        for bf in blue_flag:
            ax.plot(
                bf.col,
                bf.row,
                marker=">",
                color="mediumblue",
                markersize=flag_markersize,
            )

        for rf in red_flag:
            ax.plot(
                rf.col,
                rf.row,
                marker=">",
                color="mediumblue",
                markersize=flag_markersize,
            )

        match blue_ugv:
            case None:
                pass
            case Location(col, row):
                ax.plot(
                    col,
                    row,
                    marker="o",
                    color="royalblue",
                    markersize=agent_markersize,
                )

        match blue_uav:
            case None:
                pass
            case Location(col, row):
                ax.plot(
                    col,
                    row,
                    marker="D",
                    color="royalblue",
                    markersize=agent_markersize,
                )

        match red_ugv:
            case None:
                pass
            case Location(col, row):
                ax.plot(
                    col,
                    row,
                    marker="o",
                    color="crimson",
                    markersize=agent_markersize,
                )

        match red_uav:
            case None:
                pass
            case Location(col, row):
                ax.plot(
                    col,
                    row,
                    marker="D",
                    color="crimson",
                    markersize=agent_markersize,
                )

        image = self._get_image(fig)

        return image

    def _get_field(self) -> Field:
        return self.field
