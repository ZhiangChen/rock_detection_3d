import argparse
import numpy as np
import pandas as pd
import open3d as o3d
import laspy
from collections import deque
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import time


def load_las_as_open3d_point_cloud(las_file_path, evaluate=False):
    # Read LAS file
    pc = laspy.read(las_file_path)
    x = pc.x.scaled_array()
    y = pc.y.scaled_array()
    z = pc.z.scaled_array()

    ground_truth_labels = None
    if evaluate:
        if "Original cloud index" in pc.point_format.dimension_names:
            ground_truth_labels = np.int_(pc["Original cloud index"])
        else:
            raise ValueError(
                "The 'Original cloud index' field does not exist in the LAS file."
            )

    # Normalize coordinates
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    z_mean = np.mean(z)
    xyz = np.vstack((x - x_mean, y - y_mean, z - z_mean)).transpose()

    # Extract RGB colors if available
    if (
        "red" in pc.point_format.dimension_names
        and "green" in pc.point_format.dimension_names
        and "blue" in pc.point_format.dimension_names
    ):
        r = np.uint8(pc.red / 65535.0 * 255)
        g = np.uint8(pc.green / 65535.0 * 255)
        b = np.uint8(pc.blue / 65535.0 * 255)
        rgb = np.vstack((r, g, b)).transpose()
    else:
        rgb = np.zeros((len(x), 3))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    return pcd, ground_truth_labels


class RegionGrowingSegmentation:
    def __init__(
        self,
        pcd,
        downsample=True,
        voxel_size=0.01,
        num_neighbors=50,
        smoothness_threshold=0.99,
        distance_threshold=0.05,
        curvature_threshold=0.15,
        use_smoothness=True,
        use_curvature=True,
        rock_seeds=None,
        pedestal_seeds=None,
        basal_points=None,
        basal_proximity_threshold=0.2,  # New threshold for proximity
        basal_proximity_check=False,  # Flag to enable proximity check
    ):

        print("Downsampling pointcloud")
        if downsample:
            self.pcd = pcd.voxel_down_sample(voxel_size)  # Downsample the point cloud
        else:
            self.pcd = pcd  # No downsampling

        # Estimate normals
        radius_normal = voxel_size * 5
        self.pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius_normal, max_nn=50
            )
        )
        self.pcd.orient_normals_consistent_tangent_plane(k=10)
        self.pcd.normals = o3d.utility.Vector3dVector(-np.asarray(self.pcd.normals))

        self.num_neighbors = num_neighbors
        self.smoothness_threshold = smoothness_threshold
        self.distance_threshold = distance_threshold
        self.curvature_threshold = curvature_threshold
        self.use_smoothness = use_smoothness
        self.use_curvature = use_curvature
        self.rock_seeds = rock_seeds
        self.pedestal_seeds = pedestal_seeds
        self.basal_points = basal_points
        self.basal_proximity_threshold = basal_proximity_threshold
        self.basal_proximity_check = True if np.any(self.basal_points) else False

        self.labels = np.array([-1] * len(self.pcd.points))
        print(len(self.labels), "points after downsampling")  # -1 indicates unlabeled
        self.normals = np.asarray(self.pcd.normals)
        self.pcd_tree = o3d.geometry.KDTreeFlann(self.pcd)

    # def is_near_basal_points(self, point):
    #     if not np.any(self.basal_points):
    #         return False

    #     point = np.array(point)
    #     basal_points = np.array(self.basal_points)

    #     distances = np.linalg.norm(basal_points - point, axis=1)

    #     return np.any(distances < self.basal_proximity_threshold)

    def precompute_neighbors(self):
        neighbors = []
        for i in range(len(self.pcd.points)):
            [k, idx, _] = self.pcd_tree.search_radius_vector_3d(
                self.pcd.points[i], self.distance_threshold
            )
            neighbors.append(
                np.array(idx[1:])
            )  # Skip the first index as it's the point itself
        return neighbors

    def calculate_segmentation_criteria(self, neighbor_index, neighbors):
        neighbor_normal = self.normals[neighbor_index]
        second_order_neighbors = neighbors[neighbor_index]

        filtered_neighbors = [n for n in second_order_neighbors if self.labels[n] != -1]
        if len(filtered_neighbors) == 0:
            return float("inf")

        filtered_normals = self.normals[filtered_neighbors]
        dot_products = np.clip(np.dot(filtered_normals, neighbor_normal), -1.0, 1.0)

        return np.min(dot_products)

    def estimate_curvature(self, index):
        k, idx, _ = self.pcd_tree.search_radius_vector_3d(
            self.pcd.points[index], self.distance_threshold
        )
        if k > 1:
            neighbor_normals = self.normals[idx, :]
            curvature = np.mean(
                np.linalg.norm(
                    np.cross(neighbor_normals - self.normals[index], neighbor_normals),
                    axis=1,
                )
            )
            return curvature
        else:
            return 0

    def highlight_proximity_points(self, colored_pcd):
        basal_points = np.array(self.basal_points)
        points = np.asarray(colored_pcd.points)
        distances_to_basal = np.linalg.norm(
            basal_points[:, np.newaxis] - points, axis=2
        )
        is_near_basal = np.any(
            distances_to_basal < self.basal_proximity_threshold, axis=0
        )

        # Visualize the proximity points in a different color, e.g., green
        colors = np.asarray(colored_pcd.colors)
        colors[is_near_basal] = [0, 1, 0]  # Green for proximity
        colored_pcd.colors = o3d.utility.Vector3dVector(colors)
        return colored_pcd

    def grow_region(self, starting_queue, region_index, neighbors):
        if self.basal_proximity_check:
            print(self.basal_points)
        queue = deque(starting_queue)
        self.labels[starting_queue[0]] = region_index

        while queue:
            current_index = queue.popleft()
            current_point = np.asarray(self.pcd.points)[current_index]
            neighbor_indices = neighbors[current_index]
            neighbor_indices = neighbors[current_index]
            neighbor_points = np.asarray(self.pcd.points)[neighbor_indices]
            if self.basal_proximity_check:
                distances_to_basal = np.linalg.norm(
                    np.asarray(self.basal_points)[:, np.newaxis] - neighbor_points,
                    axis=2,
                )
                is_near_basal = np.any(
                    distances_to_basal < self.basal_proximity_threshold, axis=0
                )

                # Identify neighbors not near basal points and not yet labeled
                valid_neighbors = neighbor_indices[
                    (~is_near_basal) & (self.labels[neighbor_indices] == -1)
                ]

                # Update labels for valid neighbors and extend the queue
                self.labels[valid_neighbors] = region_index
                queue.extend(valid_neighbors)
            else:
                print("not after basal")
                for neighbor_index in neighbor_indices:
                    if self.labels[neighbor_index] != -1:
                        continue

                    min_dot_product = self.calculate_segmentation_criteria(
                        neighbor_index, neighbors
                    )
                    curvature = self.estimate_curvature(neighbor_index)

                    if (
                        self.use_smoothness
                        and min_dot_product >= self.smoothness_threshold
                    ) or (self.use_curvature and curvature < self.curvature_threshold):
                        self.labels[neighbor_index] = region_index
                        queue.append(neighbor_index)

    def segment(self):

        if self.rock_seeds is None or self.pedestal_seeds is None:
            points = np.asarray(self.pcd.points)
            min_bound = points.min(axis=0)
            max_bound = points.max(axis=0)

            centroid_x = (min_bound[0] + max_bound[0]) / 2.0
            centroid_y = (min_bound[1] + max_bound[1]) / 2.0

            distances = np.linalg.norm(
                points[:, :2] - np.array([centroid_x, centroid_y]), axis=1
            )
            highest_point_index = np.argmax(points[:, 2] - distances)
            self.rock_seeds = [highest_point_index]

            bottommost_point_index = np.argmin(points[:, 2])
            self.pedestal_seeds = [bottommost_point_index]

        neighbors = self.precompute_neighbors()

        self.grow_region(self.rock_seeds, region_index=1, neighbors=neighbors)
        self.grow_region(self.pedestal_seeds, region_index=0, neighbors=neighbors)

        return self.pcd, self.labels

    def color_point_cloud(self):
        points = np.asarray(self.pcd.points)
        colors = np.zeros_like(points)

        colors[:, :] = [0.5, 0.5, 0.5]

        colors[self.labels == 1] = [1, 0, 0]
        colors[self.labels == 0] = [0, 0, 1]

        colored_pcd = o3d.geometry.PointCloud()
        colored_pcd.points = o3d.utility.Vector3dVector(points)
        colored_pcd.colors = o3d.utility.Vector3dVector(colors)

        self.pcd.colors = o3d.utility.Vector3dVector(colors)

        return self.pcd

    def conditional_label_propagation(self, distance_threshold=0.05):
        points = np.asarray(self.pcd.points)
        tree = self.pcd_tree

        unlabeled_indices = np.where(self.labels == -1)[0]
        unchanged_iterations = 0
        prev_unlabeled_count = len(unlabeled_indices)

        while len(unlabeled_indices) > 0:
            for index in unlabeled_indices:
                [k, idx, _] = tree.search_radius_vector_3d(
                    points[index], distance_threshold
                )
                neighbor_labels = self.labels[idx]

                labeled_neighbors = neighbor_labels[neighbor_labels != -1]
                if len(labeled_neighbors) > 0:
                    self.labels[index] = np.bincount(labeled_neighbors).argmax()

            unlabeled_indices = np.where(self.labels == -1)[0]
            current_unlabeled_count = len(unlabeled_indices)

            if current_unlabeled_count == prev_unlabeled_count:
                unchanged_iterations += 1
            else:
                unchanged_iterations = 0

            if unchanged_iterations >= 2:
                break

            prev_unlabeled_count = current_unlabeled_count

        return self.labels

    def transfer_labels_to_dense(self, dense_pcd, sparse_pcd, sparse_labels):
        dense_points = np.asarray(dense_pcd.points)
        sparse_points = np.asarray(sparse_pcd.points)

        nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(sparse_points)
        distances, indices = nbrs.kneighbors(dense_points)

        dense_labels = sparse_labels[indices.flatten()]

        return dense_labels

    def evaluate_segmentation_performance(self, ground_truth_labels, predicted_labels):
        accuracy = np.mean(ground_truth_labels == predicted_labels)
        conf_matrix = pd.DataFrame(
            confusion_matrix(
                ground_truth_labels,
                predicted_labels,
            ),
            index=["true:PBR", "true:PED"],
            columns=["pred:PBR", "pred:PED"],
        )

        precision = precision_score(
            ground_truth_labels, predicted_labels, average="weighted"
        )
        recall = recall_score(
            ground_truth_labels, predicted_labels, average="weighted", zero_division=0
        )
        dice = f1_score(ground_truth_labels, predicted_labels, average="weighted")

        return accuracy, conf_matrix, precision, recall, dice


def main():
    parser = argparse.ArgumentParser(
        description="Region Growing Segmentation for Point Clouds"
    )
    parser.add_argument("las_file_path", type=str, help="Path to the input LAS file")
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.01,
        help="Voxel size for downsampling the point cloud",
    )
    parser.add_argument(
        "--distance_threshold",
        type=float,
        default=0.05,
        help="Distance threshold for region growing",
    )
    parser.add_argument(
        "--smoothness_threshold",
        type=float,
        default=0.99,
        help="Smoothness threshold for region growing",
    )
    parser.add_argument(
        "--curvature_threshold",
        type=float,
        default=0.15,
        help="Curvature threshold for region growing",
    )
    parser.add_argument(
        "--use_smoothness",
        type=bool,
        default=True,
        help="Use smoothness criteria for segmentation",
    )
    parser.add_argument(
        "--use_curvature",
        type=bool,
        default=True,
        help="Use curvature criteria for segmentation",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=False,
        help="Evaluate segmentation performance using ground truth labels",
    )

    args = parser.parse_args()

    # Load the LAS file as an Open3D point cloud and get ground truth labels
    print("Loading dense point cloud from LAS file...")
    dense_pcd, ground_truth_labels = load_las_as_open3d_point_cloud(
        args.las_file_path, evaluate=args.evaluate
    )
    print(f"Loaded point cloud with {len(dense_pcd.points)} points.")

    # Record the start time
    start_time = time.time()

    # Initialize the segmenter
    segmenter = RegionGrowingSegmentation(
        dense_pcd,
        voxel_size=args.voxel_size,
        distance_threshold=args.distance_threshold,
        smoothness_threshold=args.smoothness_threshold,
        curvature_threshold=args.curvature_threshold,
        use_smoothness=args.use_smoothness,
        use_curvature=args.use_curvature,
    )

    # Perform segmentation
    print("Initial segmentation...")
    sparse_pcd, sparse_labels = segmenter.segment()

    print("Postprocessing the segmentation...")
    segmenter.conditional_label_propagation()

    sparse_labels[sparse_labels == -1] = 1

    colored_sparse_pcd = segmenter.color_point_cloud()

    # Transfer labels to the dense point cloud
    dense_labels = segmenter.transfer_labels_to_dense(
        dense_pcd, sparse_pcd, sparse_labels
    )
    print(len(dense_labels), "points after upsampling")

    # Color the dense point cloud based on the transferred labels
    dense_colors = np.zeros_like(np.asarray(dense_pcd.points))
    dense_colors[dense_labels == 1] = [1, 0, 0]  # Red for region 0
    dense_colors[dense_labels == 0] = [0, 0, 1]  # Blue for region 1

    dense_pcd.colors = o3d.utility.Vector3dVector(dense_colors)

    # Estimate normals for visualization
    radius_normal = 0.05
    dense_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal, max_nn=50
        )
    )

    # Record the end time
    end_time = time.time()
    total_time = end_time - start_time

    print(f"Segmentation complete. Total time taken: {total_time:.2f} seconds")
    print("-" * 10)

    if args.evaluate:
        # Evaluate segmentation performance
        accuracy, conf_matrix, precision, recall, dice = (
            segmenter.evaluate_segmentation_performance(
                ground_truth_labels, dense_labels
            )
        )

        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"Dice Coefficient: {dice:.2f}")
        print("Confusion Matrix:")
        print(conf_matrix)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([dense_pcd], mesh_show_wireframe=True)


if __name__ == "__main__":
    main()
