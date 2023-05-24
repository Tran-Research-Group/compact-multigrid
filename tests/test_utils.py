import unittest
import math

import jax.numpy as jnp

from compact_multigrid.typing.field import Location
from compact_multigrid.utils import (
    distance_area_area,
    distance_points,
    distance_area_point,
)


class TestUtils(unittest.TestCase):
    def test_distance_points(self):
        p1 = Location(0, 0)
        p2 = Location(1, 1)
        self.assertAlmostEqual(distance_points(p1, p2), math.sqrt(2))

    def test_distance_area_point(self):
        p = Location(0, 0)
        area = [Location(1, 1), Location(2, 2), Location(3, 3)]
        answer = distance_area_point(p, area)

        self.assertAlmostEqual(answer, math.sqrt(2))

    def test_distance_area_area(self):
        area1 = [
            Location(0, 0),
            Location(0, 1),
            Location(1, 0),
            Location(1, 1),
        ]
        area2 = [
            Location(2, 3),
            Location(2, 2),
            Location(3, 3),
            Location(4, 4),
            Location(5, 5),
        ]

        answer = distance_area_area(area1, area2)

        self.assertAlmostEqual(answer, math.sqrt(2))
