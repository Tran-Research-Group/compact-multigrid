import jax
from jax import jit, Array
import jax.numpy as jnp
from jax.typing import ArrayLike
from pyparsing import Any

from compact_multigrid.typing.field import Location


def tuples2locs(tuples: list[tuple[int, int]]) -> list[Location]:
    locs: list[Location] = [Location(*item) for item in tuples]
    return locs


@jit
def distance_points_jax(p1: Location, p2: Location) -> float:
    """Calculate the squared distance of two points"""
    return jnp.linalg.norm(jnp.array(p1) - jnp.array(p2))


def distance_points(p1: Location, p2: Location) -> float:
    """Calculate the squared distance of two points"""
    dist = (
        distance_points_jax(jnp.array(p1), jnp.array(p2)).block_until_ready().tolist()
    )
    check_distance_type(dist)
    return dist


@jit
def distance_area_point_jax(point: Array, area: Array) -> Array:
    """Calculate the squared distance of an area and a point"""
    distance = jnp.min(jnp.linalg.norm(area - point, axis=1))
    return distance


def distance_area_point(point: Location, area: list[Location]) -> float:
    """Calculate the squared distance of an area and a point"""
    distance = (
        distance_area_point_jax(jnp.array(point), jnp.array(area))
        .block_until_ready()
        .tolist()
    )
    check_distance_type(distance)
    return distance


@jit
def distance_area_area_jax(area1: Array, area2: Array) -> Array:
    """Calculate the squared distance of an area and a point"""
    v1 = jnp.repeat(area1, area2.shape[0], axis=0)
    v2 = jnp.tile(area2, (area1.shape[0], 1))
    distance = jnp.min(jnp.linalg.norm(v1 - v2, axis=1))
    return distance


def distance_area_area(area1: list[Location], area2: list[Location]) -> float:
    distance = (
        distance_area_area_jax(jnp.array(area1), jnp.array(area2))
        .block_until_ready()
        .tolist()
    )
    check_distance_type(distance)
    return distance


def check_distance_type(distance: Any) -> None:
    match distance:
        case float():
            pass
        case _:
            raise ValueError(f"min_distance is not a float: {distance}")
