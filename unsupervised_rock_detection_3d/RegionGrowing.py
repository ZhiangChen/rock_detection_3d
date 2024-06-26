import argparse
import numpy as np
import open3d as o3d
from collections import deque
import time

class RegionGrowingSegmentation:
    def __init__(self, pcd, voxel_size=0.01, num_neighbors=50, smoothness_threshold=0.5, distance_threshold=0.05, curvature_threshold=0.5, use_smoothness=True, use_curvature=True):
        """
        Initialize the RegionGrowingSegmentation class.
        
        Parameters:
        pcd (open3d.geometry.PointCloud): The input point cloud.
        voxel_size (float): The size of voxels for downsampling the point cloud. Lower the voxel_size, better the resolution and more the computation time.
        num_neighbors (int): Number of neighboring points to consider for normal estimation.
        smoothness_threshold (float): Threshold for the dot product of normal vectors to determine smoothness.
        distance_threshold (float): Radius within which to search for neighboring points.
        curvature_threshold (float): Threshold for point curvature; lower values favor smoother regions.
        use_smoothness (bool): Flag to use smoothness criteria.
        use_curvature (bool): Flag to use curvature criteria.
        """
        
        self.pcd = pcd.voxel_down_sample(voxel_size)
        
        # Estimate normals
        radius_normal = voxel_size * 5
        self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=50))
        self.pcd.orient_normals_consistent_tangent_plane(k=10)
        self.pcd.normals = o3d.utility.Vector3dVector(-np.asarray(self.pcd.normals))
        
        self.num_neighbors = num_neighbors
        self.smoothness_threshold = smoothness_threshold
        self.distance_threshold = distance_threshold
        self.curvature_threshold = curvature_threshold
        self.use_smoothness = use_smoothness
        self.use_curvature = use_curvature
        
        self.labels = np.array([-1] * len(self.pcd.points))  # -1 indicates unlabeled
        self.normals = np.asarray(self.pcd.normals)
        self.pcd_tree = o3d.geometry.KDTreeFlann(self.pcd)

    def precompute_neighbors(self):
        """
        Precompute the neighbors for each point within the specified distance threshold.
        
        Returns:
        neighbors (list of np.array): List of neighbors for each point.
        """
        neighbors = []
        for i in range(len(self.pcd.points)):
            [k, idx, _] = self.pcd_tree.search_radius_vector_3d(self.pcd.points[i], self.distance_threshold)
            neighbors.append(np.array(idx[1:]))  # Skip the first index as it's the point itself
        return neighbors

    def calculate_segmentation_criteria(self, neighbor_index, neighbors):
        """
        Calculate the segmentation criteria for a neighbor based on the dot product of normal vectors.
        
        Parameters:
        neighbor_index (int): The index of the neighbor point.
        neighbors (list of np.array): List of neighbors for each point.
        
        Returns:
        float: The minimum dot product of normal vectors between the neighbor and its segmented neighbors.
        """
        neighbor_normal = self.normals[neighbor_index]
        second_order_neighbors = neighbors[neighbor_index]

        # Filter second-order neighbors to keep only those already segmented as PBR
        filtered_neighbors = [n for n in second_order_neighbors if self.labels[n] != -1]
        if len(filtered_neighbors) == 0:
            return float('inf')

        filtered_normals = self.normals[filtered_neighbors]
        dot_products = np.clip(np.dot(filtered_normals, neighbor_normal), -1.0, 1.0)

        return np.min(dot_products)

    def estimate_curvature(self, index):
        """
        Estimate the curvature of a point based on its neighbors.
        
        Parameters:
        index (int): The index of the point.
        
        Returns:
        float: The estimated curvature of the point.
        """
        k, idx, _ = self.pcd_tree.search_radius_vector_3d(self.pcd.points[index], self.distance_threshold)
        if k > 1:
            neighbor_normals = self.normals[idx, :]
            curvature = np.mean(np.linalg.norm(np.cross(neighbor_normals - self.normals[index], neighbor_normals), axis=1))
            return curvature
        else:
            return 0

    def grow_region(self, starting_index, region_index, neighbors):
        """
        Grow a region starting from a seed point by evaluating neighbors based on segmentation criteria.
        
        Parameters:
        starting_index (int): The index of the seed point.
        region_index (int): The index of the region to grow.
        neighbors (list of np.array): List of neighbors for each point.
        """
        queue = deque([starting_index])
        self.labels[starting_index] = region_index

        while queue:
            current_index = queue.popleft()
            current_normal = self.normals[current_index]

            neighbor_indices = neighbors[current_index]

            for neighbor_index in neighbor_indices:
                if self.labels[neighbor_index] != -1:
                    continue

                # Calculate segmentation criteria for the neighbor
                min_dot_product = self.calculate_segmentation_criteria(neighbor_index, neighbors)

                # Compute curvature for the neighbor
                curvature = self.estimate_curvature(neighbor_index)

                # Apply the thresholds based on user input
                if (self.use_smoothness and min_dot_product >= self.smoothness_threshold) or \
                   (self.use_curvature and curvature < self.curvature_threshold):
                    self.labels[neighbor_index] = region_index
                    queue.append(neighbor_index)

    def segment(self):
        """
        Perform the region growing segmentation on the point cloud.
        
        Returns:
        pcd (open3d.geometry.PointCloud): The segmented point cloud.
        labels (np.array): Array of labels for each point in the point cloud.
        """
        
        # Compute the bounding box of the point cloud
        points = np.asarray(self.pcd.points)
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)

        # Compute the centroid of the bounding box in the x and y dimensions
        centroid_x = (min_bound[0] + max_bound[0]) / 2.0
        centroid_y = (min_bound[1] + max_bound[1]) / 2.0

        # Find the point with the highest z value near the centroid in x and y dimensions
        distances = np.linalg.norm(points[:, :2] - np.array([centroid_x, centroid_y]), axis=1)
        highest_point_index = np.argmax(points[:, 2] - distances)

        # Find the bottommost corner of the bounding box
        bottommost_point_index = np.argmin(points[:, 2])

        neighbors = self.precompute_neighbors()
        
        # Grow region from the highest point near the centroid (region 0)
        self.grow_region(highest_point_index, region_index=0, neighbors=neighbors)
        
        # Grow region from the bottommost point (region 1)
        self.grow_region(bottommost_point_index, region_index=1, neighbors=neighbors)

        return self.pcd, self.labels

    def color_point_cloud(self):
        """
        Color the segmented point cloud for visualization.
        
        Returns:
        pcd (open3d.geometry.PointCloud): The colored point cloud.
        """

        points = np.asarray(self.pcd.points)
        print(len(points), "points")
        colors = np.zeros_like(points)

        # Color all points grey
        colors[:, :] = [0.5, 0.5, 0.5]  # Grey color

        # Color the labeled points
        colors[self.labels == 0] = [1, 0, 0]  # Red color for region 0
        colors[self.labels == 1] = [0, 0, 1]  # Blue color for region 1

        # Create a new point cloud with the updated colors
        colored_pcd = o3d.geometry.PointCloud()
        colored_pcd.points = o3d.utility.Vector3dVector(points)
        colored_pcd.colors = o3d.utility.Vector3dVector(colors)

        self.pcd.colors = o3d.utility.Vector3dVector(colors)

        return self.pcd


    def conditional_label_propagation(self, distance_threshold=0.05):
        """
        Assign labels to remaining unlabeled points based on majority labels of their neighbors.
        
        Parameters:
        distance_threshold (float): Radius within which to search for neighboring points.
        
        Returns:
        labels (np.array): Updated array of labels for each point in the point cloud.
        """
        
        points = np.asarray(self.pcd.points)
        tree = self.pcd_tree
        
        unlabeled_indices = np.where(self.labels == -1)[0]
        
        for index in unlabeled_indices:
            [k, idx, _] = tree.search_radius_vector_3d(points[index], distance_threshold)
            neighbor_labels = self.labels[idx]
        
            # Exclude unlabeled neighbors
            labeled_neighbors = neighbor_labels[neighbor_labels != -1]
            if len(labeled_neighbors) > 0:
                # Assign the majority label among the neighbors
                self.labels[index] = np.bincount(labeled_neighbors).argmax()
        
        return self.labels


def main():
    parser = argparse.ArgumentParser(description="Region Growing Segmentation for Point Clouds")
    parser.add_argument('pcd_path', type=str, help="Path to the input point cloud file")
    parser.add_argument('--voxel_size', type=float, default=0.01, help="Voxel size for downsampling the point cloud")
    parser.add_argument('--distance_threshold', type=float, default=0.05, help="Distance threshold for region growing")
    parser.add_argument('--smoothness_threshold', type=float, default=0.99, help="Smoothness threshold for region growing")
    parser.add_argument('--curvature_threshold', type=float, default=0.15, help="Curvature threshold for region growing")
    parser.add_argument('--use_smoothness', action='store_true', default=True, help="Use smoothness criteria for segmentation")
    parser.add_argument('--use_curvature', action='store_true', default=True, help="Use curvature criteria for segmentation")
    
    args = parser.parse_args()

    # Load the point cloud
    print("Loading Point cloud...")
    pcd = o3d.io.read_point_cloud(args.pcd_path)

    # Record the start time
    start_time = time.time()

    # Initialize the segmenter
    segmenter = RegionGrowingSegmentation(
        pcd,
        voxel_size=args.voxel_size,
        distance_threshold=args.distance_threshold,
        smoothness_threshold=args.smoothness_threshold,
        curvature_threshold=args.curvature_threshold,
        use_smoothness=args.use_smoothness,
        use_curvature=args.use_curvature
    )

    # Perform segmentation
    print("Initial segmentation...")
    pcd, labels = segmenter.segment()

    print("Postprocessing the segmentation...")
    segmenter.conditional_label_propagation()

    colored_pcd = segmenter.color_point_cloud()

    # Record the end time
    end_time = time.time()
    total_time = end_time - start_time

    print(f"Segmentation complete. Total time taken: {total_time:.2f} seconds")
    
    # Save the colored point cloud
    o3d.visualization.draw_geometries([colored_pcd],
                                  #point_show_normal = True,
                                 )

    

if __name__ == "__main__":
    main()
