import unittest

import numpy as np

from rock_detection_3d.RegionGrowing import RegionGrowingSegmentation


class _FakePointCloud:
    def __init__(self, point_count: int = 0, points=None):
        if points is None:
            self.points = np.zeros((point_count, 3), dtype=float)
        else:
            self.points = np.asarray(points, dtype=float)


class _FakeRadiusTree:
    def __init__(self, points):
        self.points = np.asarray(points, dtype=float)

    def search_radius_vector_3d(self, point, radius):
        squared_distances = np.sum((self.points - np.asarray(point, dtype=float)) ** 2, axis=1)
        indices = np.where(squared_distances <= radius * radius + 1e-12)[0]
        return len(indices), indices.tolist(), squared_distances[indices].tolist()


def _minimal_segmenter(point_count: int = 6, points=None) -> RegionGrowingSegmentation:
    segmenter = object.__new__(RegionGrowingSegmentation)
    segmenter.pcd = _FakePointCloud(point_count, points=points)
    point_count = len(segmenter.pcd.points)
    segmenter.labels = np.full(point_count, -1, dtype=int)
    segmenter.branch_ids = np.full(point_count, -1, dtype=int)
    segmenter.branch_metadata = []
    segmenter.basal_proximity_check = False
    segmenter.stepwise_visualize = False
    return segmenter


class RegionGrowingSeedConstraintTests(unittest.TestCase):
    def test_grow_region_labels_every_seed(self):
        segmenter = _minimal_segmenter()
        neighbors = [np.asarray([], dtype=int) for _ in range(6)]

        segmenter.grow_region([1, 3, 3, 5], region_index=0, neighbors=neighbors)

        np.testing.assert_array_equal(
            segmenter.labels,
            np.asarray([-1, 0, -1, 0, -1, 0], dtype=int),
        )

    def test_initialize_seed_labels_marks_all_regions_before_growth(self):
        segmenter = _minimal_segmenter()
        segmenter.rock_seeds = np.asarray([0, 2], dtype=int)
        segmenter.pedestal_seeds = np.asarray([1, 4, 4], dtype=int)

        segmenter._initialize_seed_labels()

        np.testing.assert_array_equal(
            segmenter.labels,
            np.asarray([1, 0, 1, -1, 0, -1], dtype=int),
        )
        self.assertEqual(
            [branch["label"] for branch in segmenter.branch_metadata],
            ["Rock seed 1", "Rock seed 2", "Pedestal seed 1", "Pedestal seed 2"],
        )

    def test_grown_points_inherit_seed_branch_id(self):
        segmenter = _minimal_segmenter()
        segmenter.labels[0] = 1
        segmenter.branch_ids[0] = 7
        segmenter.use_smoothness = True
        segmenter.use_curvature = False
        segmenter.smoothness_threshold = 0.5
        segmenter.calculate_segmentation_criteria = lambda _index, _neighbors: 1.0
        segmenter.estimate_curvature = lambda _index: 0.0
        neighbors = [
            np.asarray([2], dtype=int),
            np.asarray([], dtype=int),
            np.asarray([], dtype=int),
            np.asarray([], dtype=int),
            np.asarray([], dtype=int),
            np.asarray([], dtype=int),
        ]

        segmenter.grow_region([0], region_index=1, neighbors=neighbors)

        self.assertEqual(segmenter.labels[2], 1)
        self.assertEqual(segmenter.branch_ids[2], 7)

    def test_initialize_seed_labels_rejects_overlapping_seed_voxels(self):
        segmenter = _minimal_segmenter()
        segmenter.rock_seeds = np.asarray([2], dtype=int)
        segmenter.pedestal_seeds = np.asarray([2, 4], dtype=int)

        with self.assertRaisesRegex(ValueError, "overlap after voxel transfer"):
            segmenter._initialize_seed_labels()

    def test_bounded_neighbor_count_uses_requested_normal_neighbors(self):
        self.assertEqual(RegionGrowingSegmentation._bounded_neighbor_count(24, 100), 24)
        self.assertEqual(RegionGrowingSegmentation._bounded_neighbor_count(500, 12), 12)
        self.assertEqual(RegionGrowingSegmentation._bounded_neighbor_count("bad", 200), 50)

    def test_basal_constraint_presence_does_not_depend_on_coordinate_values(self):
        self.assertFalse(RegionGrowingSegmentation._has_basal_constraint_points(None))
        self.assertFalse(RegionGrowingSegmentation._has_basal_constraint_points([]))
        self.assertTrue(
            RegionGrowingSegmentation._has_basal_constraint_points(
                np.zeros((1, 3), dtype=float)
            )
        )

    def test_distance_weighted_vote_prefers_close_minority_label(self):
        selected = RegionGrowingSegmentation._distance_weighted_vote(
            np.asarray([0, 0, 1], dtype=int),
            np.asarray([10.0, 10.0, 1.0], dtype=float),
        )

        self.assertEqual(selected, 1)

    def test_distance_weighted_vote_ignores_unlabeled_values(self):
        selected = RegionGrowingSegmentation._distance_weighted_vote(
            np.asarray([-1, 2], dtype=int),
            np.asarray([0.0, 5.0], dtype=float),
        )

        self.assertEqual(selected, 2)

    def test_distance_weighted_vote_ties_by_closest_then_lowest_value(self):
        closer_selected = RegionGrowingSegmentation._distance_weighted_vote(
            np.asarray([1, 2, 2], dtype=int),
            np.asarray([1.0, 2.0, 2.0], dtype=float),
        )
        lowest_selected = RegionGrowingSegmentation._distance_weighted_vote(
            np.asarray([2, 1], dtype=int),
            np.asarray([1.0, 1.0], dtype=float),
        )

        self.assertEqual(closer_selected, 1)
        self.assertEqual(lowest_selected, 1)

    def test_radius_label_propagation_uses_distance_weighted_label_and_branch(self):
        points = np.asarray(
            [
                [0.00, 0.0, 0.0],
                [0.10, 0.0, 0.0],
                [0.06, 0.0, 0.0],
                [0.061, 0.0, 0.0],
            ],
            dtype=float,
        )
        segmenter = _minimal_segmenter(points=points)
        segmenter.labels = np.asarray([0, 0, -1, 1], dtype=int)
        segmenter.branch_ids = np.asarray([10, 10, -1, 20], dtype=int)
        segmenter.pcd_tree = _FakeRadiusTree(points)

        segmenter.conditional_label_propagation(distance_threshold=0.2)

        self.assertEqual(segmenter.labels[2], 1)
        self.assertEqual(segmenter.branch_ids[2], 20)

    def test_final_knn_fallback_uses_distance_weighted_label_and_branch(self):
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
            ],
            dtype=float,
        )
        segmenter = _minimal_segmenter(points=points)
        segmenter.labels = np.asarray([-1, 0, 0, 1], dtype=int)
        segmenter.branch_ids = np.asarray([-1, 10, 10, 20], dtype=int)
        segmenter.pcd_tree = _FakeRadiusTree(points)

        segmenter.conditional_label_propagation(distance_threshold=0.1)

        self.assertEqual(segmenter.labels[0], 1)
        self.assertEqual(segmenter.branch_ids[0], 20)


if __name__ == "__main__":
    unittest.main()
