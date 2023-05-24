from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from numpy import ndarray

from compact_multigrid.envs.ctf.base_ctf import Field
from compact_multigrid.policy.utils import a_star, closest_area_point
from compact_multigrid.typing.field import Direction, Location


class BaseProbabilisticPolicy(ABC):
    def __init__(
        self,
        field: Field,
        field_map: ndarray,
        randomness: float,
    ) -> None:
        """
        randomness: float in [0, 1]
            1: deterministic, 0: random
        """
        super().__init__()
        self._obstacles: list[Location] = field.obstacle
        self._randomness: float = randomness
        self._field_map: ndarray = field_map
        self._field: Field = field

    @abstractmethod
    def act(self) -> Location:
        ...

    def reset(self, init_loc: Location):
        self._counter: int = 0
        self._traj: list[Location] = [init_loc]
        self._init_loc: Location = init_loc
        self._current_loc: Location = init_loc

    def _get_allowed_locations(
        self, current_loc: Location, obstacles: list[Location]
    ) -> list[Location]:
        possible_locs: list[Location] = self._get_neighbors(current_loc)

        allowed_locs: list[Location] = list(set(possible_locs) - set(obstacles))

        return allowed_locs

    def _get_neighbors(self, current_loc: Location) -> list[Location]:
        directions: list[Direction] = [
            Direction(0, 1),
            Direction(0, -1),
            Direction(1, 0),
            Direction(-1, 0),
        ]

        neighbors: list[Location] = [
            Location(current_loc.row + direction.row, current_loc.col + direction.col)
            for direction in directions
        ] + [current_loc]

        return neighbors

    def _act_probabilistically(
        self,
        target: Location,
    ) -> Location:
        shortest_path: list[Location] = a_star(
            self._current_loc, target, self._field_map
        )
        optimal_loc: Location = shortest_path[1] if len(shortest_path) > 1 else target

        allowed_locs: list[Location] = self._get_allowed_locations(
            self._current_loc, self._obstacles
        )
        suboptimal_locs: list[Location] = list(set(allowed_locs) - set([optimal_loc]))
        probabilities: list[float] = [self._randomness] + [
            (1 - self._randomness) / (len(suboptimal_locs))
            for _ in range(len(suboptimal_locs))
        ]

        all_locs: list[Location] = [optimal_loc] + suboptimal_locs
        new_loc: Location = all_locs[
            np.random.choice(list(range(len(all_locs))), p=probabilities)
        ]

        return new_loc


class FightPolicy(BaseProbabilisticPolicy):
    def __init__(
        self,
        field: Field,
        field_map: np.ndarray,
        randomness: float,
    ) -> None:
        """
        randomness: float in [0, 1]
            1: deterministic, 0: random
        """
        super().__init__(field, field_map, randomness)

    def act(self, target: Location) -> Location:
        new_loc: Location = self._act_probabilistically(target)
        self._current_loc = new_loc
        self._counter += 1

        return new_loc


class PatrolPolicy(BaseProbabilisticPolicy):
    def __init__(
        self,
        field: Field,
        field_map: np.ndarray,
        randomness: float,
        agent_name: Literal["red", "blue"] = "red",
    ) -> None:
        """
        randomness: float in [0, 1]
            1: deterministic, 0: random
        """
        super().__init__(field, field_map, randomness)
        red_border, blue_border = self._locate_borders()
        self._agent_name: Literal["red", "blue"] = agent_name
        self._boarders: list[Location]
        match agent_name:
            case "red":
                self._boarders = red_border
            case "blue":
                self._boarders = blue_border
            case _:
                raise ValueError(
                    f"agent_name must be 'red' or 'blue', got {agent_name}"
                )

    def act(self) -> Location:
        allowed_locs: list[Location] = self._get_allowed_locations(
            self._current_loc, self._obstacles
        )
        optimal_locs: list[Location]
        if self._current_loc in self._boarders:
            optimal_locs = [loc for loc in allowed_locs if loc in self._boarders]
        else:
            closest_boarder_point: Location = closest_area_point(
                self._current_loc, self._boarders
            )
            shortest_path: list[Location] = a_star(
                self._current_loc, closest_boarder_point, self._field_map
            )
            optimal_locs = [shortest_path[1]]

        suboptimal_locs: list[Location] = list(set(allowed_locs) - set(optimal_locs))
        probabilities: list[float] = [
            self._randomness / len(optimal_locs) for _ in range(len(optimal_locs))
        ] + [
            (1 - self._randomness) / (len(suboptimal_locs))
            for _ in range(len(suboptimal_locs))
        ]
        all_locs: list[Location] = optimal_locs + suboptimal_locs
        new_loc: Location = all_locs[
            np.random.choice(list(range(len(all_locs))), p=probabilities)
        ]
        self._current_loc = new_loc
        self._counter += 1

        return new_loc

    def _locate_borders(self) -> tuple[list[Location], list[Location]]:
        """
        Locate the borders between the blue and red agents areas and store them in self._field.red_border and self._field.blue_border
        """
        red_border: list[Location] = []
        blue_border: list[Location] = []
        for loc in self._field.red_background:
            neighbors: list[Location] = self._get_neighbors(loc)
            for neighbor in neighbors:
                if neighbor in self._field.blue_background + self._field.obstacle:
                    red_border.append(loc)
                    break
                else:
                    pass

        for loc in self._field.blue_background:
            neighbors: list[Location] = self._get_neighbors(loc)
            for neighbor in neighbors:
                if neighbor in self._field.red_background + self._field.obstacle:
                    blue_border.append(loc)
                    break
                else:
                    pass

        return red_border, blue_border


class CapturePolicy(BaseProbabilisticPolicy):
    def __init__(
        self,
        field: Field,
        field_map: np.ndarray,
        randomness: float,
        agent_name: Literal["red", "blue"] = "red",
    ) -> None:
        super().__init__(field, field_map, randomness)
        self._agent_name: Literal["red", "blue"] = agent_name
        self._flag: list[Location]
        match agent_name:
            case "red":
                self._flag = field.blue_flag
            case "blue":
                self._flag = field.red_flag
            case _:
                raise ValueError(
                    f'agent_name must be "red" or "blue", got {agent_name}'
                )

    def act(self) -> Location:
        closest_flag: Location = closest_area_point(self._current_loc, self._flag)
        new_loc: Location = self._act_probabilistically(closest_flag)
        self._current_loc = new_loc
        self._counter += 1

        return new_loc
