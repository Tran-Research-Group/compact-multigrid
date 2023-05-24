import unittest

from compact_multigrid.envs.ctf.ctf_1v1 import Ctf1v1, EnemyPolicyMode


class TestCtf(unittest.TestCase):
    def test_render(self):
        map_path: str = "assets/map/ctf_1v1.txt"
        enemy_policy_mode: EnemyPolicyMode = "none"
        env = Ctf1v1(map_path, enemy_policy_mode, render_mode="human")
        env.reset()
        env.render(block=True)
