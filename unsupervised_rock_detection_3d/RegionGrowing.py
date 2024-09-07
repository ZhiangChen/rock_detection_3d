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
    """
    Load a LAS file and convert it to an Open3D point cloud.
    """
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
        basal_points=None,  # List or array of basal points (optional, default is None)
        basal_proximity_threshold=0.05,  # Threshold distance for checking proximity to basal points
        basal_proximity_check=False,  # Flag to override the smoothness and curvature criteria and use basal proximity
        stepwise_visualize=False,
    ):
        """
        Initialize the region growing segmentation object.
        """

        if downsample:
            print("Downsampling pointcloud")
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
        self.basal_proximity_check = (
            True if np.any(self.basal_points) else False
        )  # Enable the proximity check if basal points are provided, otherwise keep it disabled
        self.stepwise_visualize = stepwise_visualize

        self.labels = np.array([-1] * len(self.pcd.points))
        print(len(self.labels), "points after downsampling")  # -1 indicates unlabeled
        self.normals = np.asarray(self.pcd.normals)
        self.pcd_tree = o3d.geometry.KDTreeFlann(self.pcd)

    def precompute_neighbors(self):
        """
        Precompute the neighbors for each point in the point cloud based on a distance threshold.
        """
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
        """
        Calculate the smoothness criteria for a neighboring point.
        """
        # Get the normal of the current point and the normal of the neighboring point
        neighbor_normal = self.normals[neighbor_index]
        second_order_neighbors = neighbors[neighbor_index]

        # Filter out the neighbors that are not yet labeled
        filtered_neighbors = [n for n in second_order_neighbors if self.labels[n] != -1]
        if len(filtered_neighbors) == 0:
            return float("inf")

        # Calculate the dot product of the normals between the neighboring point and its labeled neighbors
        filtered_normals = self.normals[filtered_neighbors]
        dot_products = np.clip(np.dot(filtered_normals, neighbor_normal), -1.0, 1.0)

        return np.min(dot_products)

    def estimate_curvature(self, index):
        """
        Estimate the curvature at a given point.
        """
        # Retrieve the neighboring points within a certain radius
        k, idx, _ = self.pcd_tree.search_radius_vector_3d(
            self.pcd.points[index], self.distance_threshold
        )

        # Skip the first index as it's the point itself
        if k > 1:
            neighbor_normals = self.normals[idx, :]
            # Calculate the curvature based on the cross product of the normals
            curvature = np.mean(
                np.linalg.norm(
                    np.cross(neighbor_normals - self.normals[index], neighbor_normals),
                    axis=1,
                )
            )
            return curvature
        else:
            # If no neighbors are found, return 0
            return 0

    def highlight_proximity_points(self, colored_pcd):
        """
        Highlight points that are near basal points in a different color.
        """
        # Calculate the distances from each point to all basal points
        basal_points = np.array(self.basal_points)
        points = np.asarray(colored_pcd.points)
        distances_to_basal = np.linalg.norm(
            basal_points[:, np.newaxis] - points, axis=2
        )
        # Determine which points are near any basal point
        is_near_basal = np.any(
            distances_to_basal < self.basal_proximity_threshold, axis=0
        )

        # Visualize the proximity points in a different color, e.g., green
        colors = np.asarray(colored_pcd.colors)
        colors[is_near_basal] = [0, 1, 0]  # Green for proximity
        colored_pcd.colors = o3d.utility.Vector3dVector(colors)
        return colored_pcd

    def grow_region(self, starting_queue, region_index, neighbors):
        """
        Perform region growing segmentation starting from the given seeds.
        """

        # If basal proximity check is enabled, print basal points as a sanity check
        if self.basal_proximity_check:
            print(self.basal_points)

        # Initialize the queue with the starting points (seeds) for region growing
        queue = deque(starting_queue)

        # Label the initial point (seed) with the current region index
        self.labels[starting_queue[0]] = region_index

        points_marked = 0

        # Process the queue until it's empty
        while queue:

            if (self.stepwise_visualize) and (points_marked % 10000 == 0):
                print(f"Visualizing after marking {points_marked} points.")
                self.visualize_current_segmentation()

            points_marked += 1

            # Get the next point from the queue
            current_index = queue.popleft()
            current_point = np.asarray(self.pcd.points)[current_index]

            # Retrieve the indices of neighboring points
            neighbor_indices = neighbors[current_index]

            # Retrieve the coordinates of the neighboring points
            neighbor_points = np.asarray(self.pcd.points)[neighbor_indices]

            # Iterate over the neighboring points
            for neighbor_index in neighbor_indices:
                if self.labels[neighbor_index] != -1:
                    continue

                neighbor_point = np.asarray(self.pcd.points)[neighbor_index]

                # If basal proximity check is enabled, perform the proximity-based region growing
                if self.basal_proximity_check:
                    # Calculate the distances from the neighbor to all basal points
                    distances_to_basal = np.linalg.norm(
                        np.asarray(self.basal_points) - neighbor_point, axis=1
                    )

                    # Determine if this neighbor is near any basal point
                    is_near_basal = np.any(
                        distances_to_basal < self.basal_proximity_threshold
                    )

                    if is_near_basal:
                        continue  # Skip the point if it is near the basal points

                # If the point is not near basal or no basal proximity check is enabled,
                # apply the smoothness and curvature thresholds

                # Calculate the smoothness criteria (dot product of normals)
                min_dot_product = self.calculate_segmentation_criteria(
                    neighbor_index, neighbors
                )
                # Estimate the curvature for the neighboring point
                curvature = self.estimate_curvature(neighbor_index)

                # Apply the region growing criteria based on smoothness and curvature
                if (
                    self.use_smoothness and min_dot_product >= self.smoothness_threshold
                ) or (self.use_curvature and curvature < self.curvature_threshold):
                    self.labels[neighbor_index] = region_index
                    queue.append(neighbor_index)

        print(f"Points marked: {points_marked}")

    def visualize_current_segmentation(self):
        """
        Visualize the current state of the point cloud segmentation.
        """
        colored_pcd = self.color_point_cloud()
        o3d.visualization.draw_geometries([colored_pcd], mesh_show_wireframe=True)

    def segment(self):
        """
        Perform the full region growing segmentation.
        """

        if self.rock_seeds is None or self.pedestal_seeds is None:
            # Auto-select seeds if not provided
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

        # Grow the region starting from rock and pedestal seeds
        self.grow_region(self.rock_seeds, region_index=1, neighbors=neighbors)
        self.grow_region(self.pedestal_seeds, region_index=0, neighbors=neighbors)

        return self.pcd, self.labels

    def color_point_cloud(self):
        """
        Color the segmented point cloud based on the labels.
        """

        points = np.asarray(self.pcd.points)
        colors = np.zeros_like(points)

        # Default color (gray)
        colors[:, :] = [0.5, 0.5, 0.5]

        # Color rock region red and pedestal region blue
        colors[self.labels == 1] = [1, 0, 0]
        colors[self.labels == 0] = [0, 0, 1]

        colored_pcd = o3d.geometry.PointCloud()
        colored_pcd.points = o3d.utility.Vector3dVector(points)
        colored_pcd.colors = o3d.utility.Vector3dVector(colors)

        # Update the point cloud colors
        self.pcd.colors = o3d.utility.Vector3dVector(colors)

        return self.pcd

    def conditional_label_propagation(self, distance_threshold=0.05):
        """
        Propagate labels to unlabeled points based on the labels of nearby points.
        """
        points = np.asarray(self.pcd.points)
        tree = self.pcd_tree

        unlabeled_indices = np.where(self.labels == -1)[0]
        # Perform label propagation until no more points are labeled
        unchanged_iterations = 0
        prev_unlabeled_count = len(unlabeled_indices)

        while len(unlabeled_indices) > 0:
            for index in unlabeled_indices:
                # Find the neighbors within a certain radius
                [k, idx, _] = tree.search_radius_vector_3d(
                    points[index], distance_threshold
                )
                neighbor_labels = self.labels[idx]
                # Filter out the unlabeled neighbors
                labeled_neighbors = neighbor_labels[neighbor_labels != -1]
                if len(labeled_neighbors) > 0:
                    # Assign the most common label among the neighbors
                    self.labels[index] = np.bincount(labeled_neighbors).argmax()

            unlabeled_indices = np.where(self.labels == -1)[0]
            current_unlabeled_count = len(unlabeled_indices)

            # Check for convergence
            if current_unlabeled_count == prev_unlabeled_count:
                unchanged_iterations += 1
            else:
                unchanged_iterations = 0

            # Break if no more points are labeled or the number of unlabeled points is unchanged
            if unchanged_iterations >= 2:
                break

            prev_unlabeled_count = current_unlabeled_count

        return self.labels

    def transfer_labels_to_dense(self, dense_pcd, sparse_pcd, sparse_labels):
        """
        Transfer labels from a sparse point cloud to a dense point cloud.
        """
        dense_points = np.asarray(dense_pcd.points)
        sparse_points = np.asarray(sparse_pcd.points)

        # Find the nearest neighbor in the sparse point cloud for each point in the dense point cloud
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(sparse_points)
        distances, indices = nbrs.kneighbors(dense_points)

        dense_labels = sparse_labels[indices.flatten()]

        return dense_labels

    def evaluate_segmentation_performance(self, ground_truth_labels, predicted_labels):
        """
        Evaluate the segmentation performance using various metrics.
        """
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
