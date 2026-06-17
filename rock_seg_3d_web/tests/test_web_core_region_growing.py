import unittest

import numpy as np

from rock_seg_3d_web.core.RegionGrowing import RegionGrowingSegmentation


class WebCoreRegionGrowingTests(unittest.TestCase):
    def test_copied_core_uses_distance_weighted_vote(self):
        selected = RegionGrowingSegmentation._distance_weighted_vote(
            values=np.asarray([1, 0, 0], dtype=int),
            distances=np.asarray([0.05, 0.2, 0.21], dtype=float),
        )

        self.assertEqual(selected, 1)

    def test_copied_core_bounds_neighbor_count(self):
        self.assertEqual(RegionGrowingSegmentation._bounded_neighbor_count(24, 100), 24)
        self.assertEqual(RegionGrowingSegmentation._bounded_neighbor_count(500, 12), 12)
        self.assertEqual(RegionGrowingSegmentation._bounded_neighbor_count("bad", 200), 50)


if __name__ == "__main__":
    unittest.main()
