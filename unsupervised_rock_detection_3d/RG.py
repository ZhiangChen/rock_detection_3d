import numpy as np
import open3d as o3d
from collections import deque
import matplotlib.pyplot as plt

class RegionGrowingSegmentation:
    def __init__(self, pcd, num_neighbors=50, smoothness_threshold=0.95, distance_threshold=0.1, curvature_threshold = 0.5):
        voxel_size = 0.05
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
        
        
        self.labels = np.array([-1] * len(self.pcd.points))  # -1 indicates unlabeled
        self.normals = np.asarray(self.pcd.normals)
        self.pcd_tree = o3d.geometry.KDTreeFlann(self.pcd)

    def precompute_neighbors(self):
        neighbors = []
        for i in range(len(self.pcd.points)):
            [k, idx, _] = self.pcd_tree.search_radius_vector_3d(self.pcd.points[i], self.distance_threshold)
            neighbors.append(np.array(idx[1:]))  # Skip the first index as it's the point itself
        return neighbors

    def grow_region(self, starting_indices, neighbors):
        for region_index, starting_index in enumerate(starting_indices):
            if self.labels[starting_index] == -1:  # If not labeled
                queue = deque([starting_index])
                self.labels[starting_index] = region_index
        
                while queue:
                    current_index = queue.popleft()
                    current_normal = self.normals[current_index]
        
                    neighbor_indices = neighbors[current_index]
                    neighbor_normals = self.normals[neighbor_indices]
                    dot_products = np.clip(np.dot(neighbor_normals, current_normal), -1.0, 1.0)
        
                    curvatures = np.linalg.norm(np.cross(neighbor_normals - current_normal,neighbor_normals), axis=1)
                    
#                    valid_indices = neighbor_indices[(curvatures<self.curvature_threshold) & (self.labels[neighbor_indices] == -1)]
                    valid_indices = neighbor_indices[(dot_products >= self.smoothness_threshold) & (curvatures<self.curvature_threshold) & (self.labels[neighbor_indices] == -1)]
    
                    
                    self.labels[valid_indices] = region_index
                    queue.extend(valid_indices)

    def segment(self):
        # Choose the lowest point as terrain seed
        z_values = np.array(self.pcd.points)[:, 2]
        terrain_seed = np.argmin(z_values)

        # Choose a high point with potentially fewer neighbors as rock seed
        potential_rock_seeds = np.where(z_values > np.percentile(z_values, 80))[0]
        rock_seed = potential_rock_seeds[np.argmax(z_values[potential_rock_seeds])]
        
        neighbors = self.precompute_neighbors()
        self.grow_region([terrain_seed, rock_seed], neighbors)

        # Assign remaining unlabeled points to the nearest labeled point
        # unlabeled_points = np.where(self.labels == -1)[0]
        # for unlabeled_point in unlabeled_points:
        #     [_, idx, _] = self.pcd_tree.search_knn_vector_3d(self.pcd.points[unlabeled_point], 1)
        #     self.labels[unlabeled_point] = self.labels[idx[0]]

        return self.pcd, self.labels

    def visualize_segmentation(self):
        max_label = self.labels.max()
        colors = plt.get_cmap('viridis')(self.labels / (max_label if max_label > 0 else 1))
        colors = colors[:, :3]  # remove the alpha channel
        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([self.pcd],
                                          #point_show_normal = True
                                         )



# Load point cloud
pcd = o3d.io.read_point_cloud("box_pbr/pbr28.pcd")

# Create an instance of the RegionGrowingSegmentation class
rgs = RegionGrowingSegmentation(pcd, distance_threshold=0.1, smoothness_threshold=0.95, curvature_threshold= 0.54)

# Perform region growing segmentation
pcd, segment_labels = rgs.segment()

rgs.visualize_segmentation()