from heapq import heapify, heappop, heappush
from typing import NamedTuple, Union

import numpy as np
from compact_multigrid.typing.field import Direction, Location
from compact_multigrid.utils import manhattan_distance


class AStarNode(NamedTuple):
    f: int
    g: int
    h: int
    parent: Union["AStarNode", None]
    loc: Location


def a_star(
    start: Location, end: Location, map: np.ndarray, obstacle_id: int = 8
) -> list[Location]:
    """
    Compute the path-planning for the red agent using A* algorithm

    Parameters:
    start (Location): start location
    end (Location): goal location

    Returns:
    path (list[Location]): the path from its original locition to the goal.
    """
    rows, cols = map.shape
    map_list: list[list[float]] = map.tolist()
    # Add the start and end nodes
    start_node = AStarNode(
        manhattan_distance(start, end), 0, manhattan_distance(start, end), None, start
    )
    # Initialize and heapify the lists
    open_nodes: list[AStarNode] = [start_node]
    closed_nodes: list[AStarNode] = []
    heapify(open_nodes)
    path: list[Location] = []  # return of the func

    while open_nodes:
        # Get the current node popped from the open list
        current_node = heappop(open_nodes)

        # Push the current node to the closed list
        closed_nodes.append(current_node)

        # When the goal is found
        if current_node.loc == end:
            current: AStarNode | None = current_node
            while current is not None:
                path.append(current.loc)
                current = current.parent

            path.reverse()
            break

        else:
            for direction in [
                Direction(0, 1),
                Direction(0, -1),
                Direction(1, 0),
                Direction(-1, 0),
            ]:
                # Get node location
                current_loc: Location = current_node.loc
                new_loc = Location(
                    current_loc.row + direction.row, current_loc.col + direction.col
                )

                # Make sure within a range
                if (
                    (new_loc.col >= 0 and new_loc.col < cols)
                    and (new_loc.row >= 0 and new_loc.row < rows)
                    and (map_list[new_loc.row][new_loc.col] != obstacle_id)
                ):
                    # Create the f, g, and h values
                    g = current_node.g + 1
                    h = manhattan_distance(new_loc, end)
                    f = g + h

                    # Check if the new node is in the open or closed list
                    open_indices = [
                        i
                        for i, open_node in enumerate(open_nodes)
                        if open_node.loc == new_loc
                    ]
                    closed_indices = [
                        i
                        for i, closed_node in enumerate(closed_nodes)
                        if closed_node.loc == new_loc
                    ]

                    # Compare f values if the new node is already existing in either list
                    if closed_indices:
                        closed_index = closed_indices[0]
                        if f < closed_nodes[closed_index].f:
                            closed_nodes.pop(closed_index)
                            heappush(
                                open_nodes, AStarNode(f, g, h, current_node, new_loc)
                            )
                        else:
                            continue

                    elif open_indices:
                        open_index = open_indices[0]
                        if f < open_nodes[open_index].f:
                            open_nodes.pop(open_index)
                            open_nodes.append(AStarNode(f, g, h, current_node, new_loc))
                            heapify(open_nodes)
                        else:
                            continue

                    else:
                        heappush(open_nodes, AStarNode(f, g, h, current_node, new_loc))

                else:
                    continue

    return path


def closest_area_point(point: Location, area: list[Location]) -> Location:
    """Calculate the squared distance of an area and a point"""
    distances = [np.linalg.norm(np.array(point) - np.array(node)) for node in area]
    return area[np.argmin(distances)]
