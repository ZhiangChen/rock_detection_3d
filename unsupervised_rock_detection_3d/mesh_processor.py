import open3d as o3d
import numpy as np
import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List, Union
from geomdl import BSpline
from geomdl import utilities
from scipy.spatial import cKDTree
import traceback
import os
from sklearn.cluster import DBSCAN
# import pymeshlab

from visualization import PointCloudVisualization

class MeshProcessor:
    """
    Handles all mesh-related operations including mesh reconstruction,
    bottom face generation, and mesh saving.
    """
    
    def __init__(self):
        self.temp_mesh_path = None
        self.reconstructed_mesh = None

    def clean_outliers_dbscan(self, pcd: o3d.geometry.PointCloud, 
                             eps: float = 0.05, 
                             min_samples: int = 50,
                             return_inlier_indices: bool = False) -> Union[o3d.geometry.PointCloud, Tuple[o3d.geometry.PointCloud, np.ndarray]]:
        """
        Remove outliers using DBSCAN clustering algorithm.
        
        Args:
            pcd: Input point cloud
            eps: Maximum distance between points in the same cluster
            min_samples: Minimum number of points to form a dense region
            return_inlier_indices: Whether to return indices of inlier points
            
        Returns:
            Cleaned point cloud and optionally indices of inlier points
        """
        try:
            logging.info("Removing outliers using DBSCAN clustering...")
            points = np.asarray(pcd.points)
            
            # Skip if too few points
            if len(points) < min_samples:
                logging.warning(f"Too few points ({len(points)}) to perform DBSCAN. Skipping outlier removal.")
                if return_inlier_indices:
                    return pcd, np.arange(len(points))
                return pcd
                
            # Apply DBSCAN clustering
            db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(points)
            labels = db.labels_
            
            # Count points in each cluster (including noise as -1)
            unique_labels, counts = np.unique(labels, return_counts=True)
            cluster_count = len(unique_labels) - (1 if -1 in unique_labels else 0)
            logging.info(f"DBSCAN found {cluster_count} clusters")
            
            if -1 in unique_labels:
                noise_idx = np.where(unique_labels == -1)[0][0]
                noise_count = counts[noise_idx]
                outlier_percentage = noise_count/len(points)*100
                logging.info(f"DBSCAN identified {noise_count} outliers ({outlier_percentage:.2f}% of points)")
                
                # Add more detailed information about the outlier detection
                if outlier_percentage < 0.5:
                    logging.warning("Very few outliers detected (<0.5%). Consider using more aggressive parameters.")
            else:
                logging.warning("No outliers identified. All points belong to clusters.")
                logging.info("Try decreasing 'eps' or increasing 'min_samples' for more aggressive outlier removal.")
            
            # Check if too many points would be removed (>50%)
            if np.sum(labels == -1) > len(points) * 0.7:
                logging.warning("DBSCAN would remove >50% of points. Adjusting parameters...")
                # Try with more lenient parameters
                return self.clean_outliers_dbscan(pcd, eps=eps*1.5, min_samples=max(5, min_samples//2), 
                                                return_inlier_indices=return_inlier_indices)
            
            # Get indices of inlier points (not labeled as noise)
            inlier_indices = np.where(labels != -1)[0]
            
            # Create cleaned point cloud
            cleaned_pcd = o3d.geometry.PointCloud()
            cleaned_pcd.points = o3d.utility.Vector3dVector(points[inlier_indices])
            
            # Copy colors and normals if they exist
            if pcd.has_colors():
                cleaned_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[inlier_indices])
            
            if pcd.has_normals():
                cleaned_pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[inlier_indices])
            
            logging.info(f"Removed {len(points) - len(inlier_indices)} outliers. Kept {len(inlier_indices)} points.")
            
            if return_inlier_indices:
                return cleaned_pcd, inlier_indices
            return cleaned_pcd
            
        except Exception as e:
            logging.error(f"Error in DBSCAN outlier removal: {str(e)}\n{traceback.format_exc()}")
            logging.warning("Continuing with original point cloud without outlier removal")
            if return_inlier_indices:
                return pcd, np.arange(len(points))
            return pcd

    def reconstruct_mesh(self, pcd: o3d.geometry.PointCloud, labels: np.ndarray, 
                        basal_points: np.ndarray, dense_basal_parts: list = None, 
                        dense_basal_parts_is_lateral: list = None, degree_u: int = 4, degree_v: int = 4,
                        control_points_u: int = 5, control_points_v: int = 5, use_dbscan_cleaning: bool = False,
                        depth: int = 8, debug_mode: bool = False, 
                        intermediate_visualization: bool = False) -> o3d.geometry.TriangleMesh:
        """
        Reconstructs a 3D mesh from the segmented point cloud.
        Process:
        1. Filters points to keep rock and basal points
        2. Optionally removes outliers using DBSCAN
        3. Generates bottom face using NURBS interpolation
        4. Performs Poisson reconstruction (Open3D or PyMeshLab)
        
        Args:
            pcd: Open3D PointCloud object
            labels: Array of point labels (0 for pedestal, 1 for rock)
            basal_points: Array of basal point indices or boolean mask
            dense_basal_parts: List of dense basal parts
            dense_basal_parts_is_lateral: List of boolean flags indicating which parts are lateral
            degree_u, degree_v: Degrees for NURBS surface
            control_points_u, control_points_v: Number of control points for NURBS surface
            use_dbscan_cleaning: Whether to use DBSCAN for outlier removal
            depth: The maximum depth of the octree used for Poisson reconstruction
            debug_mode: Whether to show debug visualizations (for test scripts)
            intermediate_visualization: Whether to show intermediate visualization and return early
            
        Returns:
            o3d.geometry.TriangleMesh: Reconstructed mesh, or tuple if intermediate_visualization is True
        """
        try:
            # Get the points and labels
            points = np.asarray(pcd.points)
            
            # Create a boolean mask for rock points
            rock_points = labels == 1
            
            # Create a boolean mask for basal points with the same shape as rock_points
            basal_mask = np.zeros_like(rock_points, dtype=bool)
            if isinstance(basal_points, np.ndarray) and len(basal_points.shape) == 1:
                # If basal_points is indices
                basal_mask[basal_points] = True
            else:
                # If basal_points is already a boolean mask
                basal_mask = basal_points

            # Combine masks
            filtered_indices = np.logical_or(rock_points, basal_mask)
            filtered_points = points[filtered_indices]
            filtered_colors = np.asarray(pcd.colors)[filtered_indices]

            # Create filtered point cloud
            filtered_pcd = o3d.geometry.PointCloud()
            filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
            filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

            # Clean outliers using DBSCAN clustering if enabled
            if use_dbscan_cleaning:
                filtered_pcd, inlier_indices = self.clean_outliers_dbscan(
                    filtered_pcd, 
                    eps=0.05,  # Adjust based on point cloud density
                    min_samples=10,
                    return_inlier_indices=True
                )
                
                # Update indices for basal points in clean point cloud
                if len(inlier_indices) < len(filtered_points):
                    # Create a mapping from original indices to new indices
                    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(inlier_indices)}
                    
                    # Get indices for basal points in filtered point cloud
                    basal_indices_in_filtered = np.where(basal_mask[filtered_indices])[0]
                    
                    # Map to new indices in cleaned point cloud
                    basal_indices = np.array([old_to_new[idx] for idx in basal_indices_in_filtered 
                                            if idx in old_to_new])
                    
                    # Get indices for rock points in filtered point cloud
                    rock_indices_in_filtered = np.where(rock_points[filtered_indices])[0]
                    
                    # Map to new indices in cleaned point cloud
                    rock_indices = np.array([old_to_new[idx] for idx in rock_indices_in_filtered 
                                        if idx in old_to_new])
                else:
                    # Get indices for basal points in filtered point cloud
                    basal_indices = np.where(basal_mask[filtered_indices])[0]
                    rock_indices = np.where(rock_points[filtered_indices])[0]
            else:
                # Skip DBSCAN cleaning, just use the filtered indices directly
                basal_indices = np.where(basal_mask[filtered_indices])[0]
                rock_indices = np.where(rock_points[filtered_indices])[0]

            # Generate bottom face points
            bottom_points = self.generate_bottom_face_points(
                filtered_pcd,
                basal_indices,
                dense_basal_parts=dense_basal_parts,
                dense_basal_parts_is_lateral=dense_basal_parts_is_lateral,
                degree_u=degree_u,
                degree_v=degree_v,
                control_points_u=control_points_u,
                control_points_v=control_points_v
            )
            
            if bottom_points is None:
                logging.error("Failed to generate bottom face points")
            
            # Process rock points
            rock_points = np.asarray(filtered_pcd.points)[rock_indices]
            rock_pcd = o3d.geometry.PointCloud()
            rock_pcd.points = o3d.utility.Vector3dVector(rock_points)
            
            # Estimate normals with better parameters for rock points
            rock_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            rock_pcd.orient_normals_consistent_tangent_plane(100, 10)
            
            # Process bottom points
            bottom_pcd = o3d.geometry.PointCloud()
            bottom_pcd.points = o3d.utility.Vector3dVector(bottom_points)

            # # Add a small gap by moving bottom points down
            # bottom_points_array = np.asarray(bottom_pcd.points)
            # gap_size = 0.02  # Adjust gap size (meters)
            # bottom_points_array[:, 2] -= gap_size  # Shift Z coordinates down
            # bottom_pcd.points = o3d.utility.Vector3dVector(bottom_points_array)
            
            bottom_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            
            # Reorient normals for rock_pcd
            logging.info("Reorienting normals for rock_pcd...")
            rock_center = rock_pcd.get_center()
            rock_points = np.asarray(rock_pcd.points)
            rock_normals = np.asarray(rock_pcd.normals)
            rock_directions = rock_points - rock_center
            rock_directions = rock_directions / np.linalg.norm(rock_directions, axis=1)[:, np.newaxis]
            rock_dots = np.sum(rock_directions * rock_normals, axis=1)
            rock_normals[rock_dots < 0] = -rock_normals[rock_dots < 0]
            rock_pcd.normals = o3d.utility.Vector3dVector(rock_normals)

            # Reorient normals for bottom_pcd
            logging.info("Reorienting normals for bottom_pcd...")
            bottom_center = bottom_pcd.get_center()
            bottom_points = np.asarray(bottom_pcd.points)
            bottom_normals = np.asarray(bottom_pcd.normals)
            bottom_directions = bottom_points - bottom_center
            bottom_directions = bottom_directions / np.linalg.norm(bottom_directions, axis=1)[:, np.newaxis]
            bottom_dots = np.sum(bottom_directions * bottom_normals, axis=1)
            bottom_normals[bottom_dots < 0] = -bottom_normals[bottom_dots < 0]
            bottom_pcd.normals = o3d.utility.Vector3dVector(bottom_normals)
            bottom_pcd.orient_normals_consistent_tangent_plane(100, 10)

            # Visualize the rock points with bottom face points before SOR filter (only in debug mode)
            if debug_mode:
                logging.info("Visualizing rock points with bottom face points before SOR filter...")
                visualizer = PointCloudVisualization()
                combined_points_pre_sor = np.vstack((np.asarray(rock_pcd.points), np.asarray(bottom_pcd.points)))
                combined_colors_pre_sor = np.vstack((
                    np.full((len(np.asarray(rock_pcd.points)), 3), [1.0, 0.0, 0.0]),  # Red for rock points
                    np.full((len(np.asarray(bottom_pcd.points)), 3), [0.0, 1.0, 0.0])  # Green for bottom face points
                ))
                visualizer.show_point_cloud(combined_points_pre_sor, combined_colors_pre_sor, "Before SOR Filter")

            # Apply SOR filter to the rock points
            logging.info("Applying SOR filter to rock points...")
            rock_pcd, inlier_indices = rock_pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=2.0)

            # Visualize the rock points with bottom face points after SOR filter
            logging.info("Visualizing rock points with bottom face points after SOR filter...")
            combined_points_post_sor = np.vstack((np.asarray(rock_pcd.points), np.asarray(bottom_pcd.points)))
            combined_colors_post_sor = np.vstack((
                np.full((len(np.asarray(rock_pcd.points)), 3), [1.0, 0.0, 0.0]),  # Red for rock points
                np.full((len(np.asarray(bottom_pcd.points)), 3), [0.0, 1.0, 0.0])  # Green for bottom face points
            ))
            
            # If intermediate visualization is requested, show it and return the prepared data
            if intermediate_visualization:
                if debug_mode:
                    # Only show visualization directly in debug mode (test scripts)
                    visualizer = PointCloudVisualization()
                    visualizer.show_point_cloud(combined_points_post_sor, combined_colors_post_sor, "After SOR Filter")
                # Return the prepared point clouds for final reconstruction
                return (rock_pcd, bottom_pcd, combined_points_post_sor, combined_colors_post_sor)

            # Combine points and normals
            combined_points = np.vstack((
                np.asarray(rock_pcd.points),
                np.asarray(bottom_pcd.points)
            ))
            combined_normals = np.vstack((
                np.asarray(rock_pcd.normals),
                np.asarray(bottom_pcd.normals)
            ))

            # Create final point cloud
            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(combined_points)
            new_pcd.normals = o3d.utility.Vector3dVector(combined_normals)

            # Color the points
            new_pcd.paint_uniform_color([1, 0, 0])

            logging.info("Using Open3D Poisson reconstruction...")
            self.reconstructed_mesh = self.poisson_reconstruction(
                new_pcd,
                depth=depth,        # Use the provided depth parameter
                width=0.0,          # Added density filtering
                scale=1.5,          # Slightly larger scale
                linear_fit=False    # Use False as in the test file
            )
            
            # Post-process the mesh (only if using Open3D, PyMeshLab method already does this)
            self.reconstructed_mesh.remove_degenerate_triangles()
            self.reconstructed_mesh.remove_duplicated_triangles()
            self.reconstructed_mesh.remove_duplicated_vertices()
            self.reconstructed_mesh.remove_non_manifold_edges()
        
            # Save temporary mesh for visualization
            with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as temp_file:
                self.temp_mesh_path = temp_file.name
            o3d.io.write_triangle_mesh(self.temp_mesh_path, self.reconstructed_mesh)

            return self.reconstructed_mesh, new_pcd

        except Exception as e:
            logging.error(f"Error in mesh reconstruction: {str(e)}")
            raise

    @staticmethod
    def poisson_reconstruction(pcd: o3d.geometry.PointCloud, 
                             depth: int = 8, 
                             width: int = 0,
                             scale: float = 1.1,
                             linear_fit: bool = False) -> o3d.geometry.TriangleMesh:
        """
        Performs Poisson surface reconstruction with improved edge handling.
        """
        try:
            # Ensure we have enough points for reconstruction
            if len(pcd.points) < 100:
                raise ValueError("Not enough points for reconstruction")

            # Perform Poisson reconstruction with optimized parameters
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd,
                depth=depth,          # Keep depth high for detail
                width=width,          
                scale=scale,          # Slightly larger scale to ensure closure
                linear_fit=linear_fit
            )

            # Clean up mesh
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()

            return mesh

        except Exception as e:
            logging.error(f"Error in Poisson reconstruction: {str(e)}")
            raise

    def generate_bottom_face_points(self, pcd: o3d.geometry.PointCloud, 
                                  basal_indices: np.ndarray,
                                  dense_basal_parts: list = None,
                                  dense_basal_parts_is_lateral: list = None,
                                  degree_u: int = 3, 
                                  degree_v: int = 3,
                                  control_points_u: int = 10,
                                  control_points_v: int = 10 ) -> np.ndarray:
        """
        Generate bottom face points with support for multiple parts
        
        Args:
            pcd: Open3D PointCloud object
            basal_indices: Array of indices for basal points
            dense_basal_parts: List of dense basal parts
            dense_basal_parts_is_lateral: List of boolean flags indicating which parts are lateral
            degree_u, degree_v: Degrees for NURBS surface
            control_points_u, control_points_v: Number of control points
            
        Returns:
            np.ndarray: Generated bottom face points
        """
        try:
            logging.info("Starting bottom face generation")
            
            # Check if we're dealing with multiple parts
            if dense_basal_parts:
                return self._generate_multi_part_faces(
                    pcd, 
                    dense_basal_parts,
                    dense_basal_parts_is_lateral,
                    degree_u, 
                    degree_v,
                    control_points_u,
                    control_points_v
                )
            else:
                return self._generate_single_face(
                    pcd,
                    basal_indices,
                    degree_u,
                    degree_v,
                    control_points_u,
                    control_points_v
                )
                
        except Exception as e:
            logging.error(f"Error in bottom face generation: {str(e)}")
            return None

    def _generate_multi_part_faces(self, pcd: o3d.geometry.PointCloud,
                                 dense_basal_parts: list,
                                 dense_basal_parts_is_lateral: list,
                                 degree_u: int,
                                 degree_v: int,
                                 control_points_u: int,
                                 control_points_v: int) -> np.ndarray:
        """
        Generate separate faces for each basal part using dense basal points
        
        Args:
            pcd: Open3D PointCloud object
            dense_basal_parts: List of dense basal points for each part
            dense_basal_parts_is_lateral: List of boolean flags indicating which parts are lateral
            degree_u, degree_v: Degrees for NURBS surface
            control_points_u, control_points_v: Number of control points
            
        Returns:
            np.ndarray: Combined face points from all parts
        """
        try:
            all_face_points = []
            # Convert Open3D points to numpy array
            points = np.asarray(pcd.points)
            
            # Use the dense basal parts that were generated and stored earlier
            if dense_basal_parts is None or not dense_basal_parts:
                logging.error("No dense basal parts found")
                return None
            
            # Generate distinct colors for visualization
            part_colors = [
                [1, 0, 0],    # Red
                [0, 1, 0],    # Green
                [0, 0, 1],    # Blue
                [1, 1, 0],    # Yellow
                [1, 0, 1],    # Magenta
            ]
            
            for i, dense_points in enumerate(dense_basal_parts):
                is_lateral = dense_basal_parts_is_lateral[i] if dense_basal_parts_is_lateral and i < len(dense_basal_parts_is_lateral) else False
                part_type = "lateral" if is_lateral else "basal"
                logging.info(f"Processing dense part {i+1}/{len(dense_basal_parts)} ({part_type})")
                try:
                    # Adjust control points based on part size
                    part_size = len(dense_points)
                    adjusted_control_u = min(control_points_u, max(4, part_size // 2))
                    adjusted_control_v = min(control_points_v, max(4, part_size // 3))
                    logging.debug(f"Adjusted control points for part {i+1}: {adjusted_control_u}x{adjusted_control_v}")
                    
                    # Generate face for this part
                    face_points = self._generate_single_face(
                        pcd, 
                        dense_points,
                        degree_u,
                        degree_v,
                        adjusted_control_u,
                        adjusted_control_v,
                        is_dense_points=True  # New flag to indicate we're passing dense points
                    )
                    
                    if face_points is not None:
                        all_face_points.append(face_points)
                        logging.info(f"Generated {len(face_points)} points for part {i+1}")
                    else:
                        logging.warning(f"Failed to generate face for part {i+1}")
                    
                except Exception as e:
                    logging.error(f"Error processing part {i+1}: {str(e)}")
                    continue
            
            # Combine all generated points
            if all_face_points:
                combined_points = np.vstack(all_face_points)
                logging.info(f"Combined {len(combined_points)} points from all parts")
                return combined_points
            return None
            
        except Exception as e:
            logging.error(f"Error in multi-part face generation: {str(e)}\n{traceback.format_exc()}")
            return None

    def _generate_single_face(self, pcd: o3d.geometry.PointCloud,
                            points_or_indices: np.ndarray,
                            degree_u: int,
                            degree_v: int,
                            control_points_u: int,
                            control_points_v: int,
                            is_dense_points: bool = False) -> np.ndarray:
        """
        Generate bottom face for a single part
        
        Args:
            pcd: Open3D PointCloud object
            points_or_indices: Array of point indices or dense points
            degree_u, degree_v: Degrees for NURBS surface
            control_points_u, control_points_v: Number of control points
            is_dense_points: Whether points_or_indices contains actual points
            
        Returns:
            np.ndarray: Generated face points
        """
        try:
            logging.debug(f"Starting bottom face generation with {len(points_or_indices)} points")
            points = np.asarray(pcd.points)
            
            # Handle input points based on whether they're dense points or indices
            if is_dense_points:
                basal_points = points_or_indices  # Already points
            else:
                basal_points = points[points_or_indices]  # Convert indices to points
            
            # Calculate transformation matrix for 2D projection
            center = np.mean(basal_points, axis=0)
            centered_points = basal_points - center
            logging.debug(f"Centered points shape: {centered_points.shape}")
            
            U, S, Vh = np.linalg.svd(centered_points)
            normal = Vh[2]
            
            # Create transformation matrix
            u = np.cross(normal, [0, 0, 1])
            if np.linalg.norm(u) < 1e-6:
                u = np.cross(normal, [0, 1, 0])
            u = u / np.linalg.norm(u)
            v = np.cross(normal, u)
            transform_matrix = np.vstack((u, v)).T
            
            logging.debug(f"Transform matrix shape: {transform_matrix.shape}")

            def create_boundary_grid(basal_points, num_u, num_v):
                """Create a grid that maintains connection with basal points"""
                try:
                    logging.debug(f"Creating boundary grid with dimensions: {num_u}x{num_v}")
                    logging.debug(f"Input basal points shape: {basal_points.shape}")
                    
                    # Project points to 2D
                    points_2d = np.dot(basal_points - center, transform_matrix)
                    logging.debug(f"2D projected points shape: {points_2d.shape}")
                    
                    # Calculate bounds with some padding
                    x_min, x_max = np.min(points_2d[:, 0]), np.max(points_2d[:, 0])
                    y_min, y_max = np.min(points_2d[:, 1]), np.max(points_2d[:, 1])
                    
                    # Add padding to ensure coverage
                    padding = 0.1 * max(x_max - x_min, y_max - y_min)
                    x_min -= padding
                    x_max += padding
                    y_min -= padding
                    y_max += padding
                    
                    # Create regular grid
                    x_grid = np.linspace(x_min, x_max, num_u)
                    y_grid = np.linspace(y_min, y_max, num_v)
                    xx, yy = np.meshgrid(x_grid, y_grid)
                    
                    # Initialize grid points
                    grid_points = np.zeros((num_v, num_u, 3))
                    
                    # Build KD-tree for nearest neighbor search
                    tree = cKDTree(points_2d)
                    
                    # For each grid point, interpolate height from nearest basal points
                    for i in range(num_v):
                        for j in range(num_u):
                            x, y = xx[i, j], yy[i, j]
                            query_point = np.array([x, y])
                            
                            # Find nearest neighbors
                            k = min(4, len(points_2d))  # Use fewer neighbors for more local influence
                            distances, indices = tree.query(query_point, k=k)
                            
                            # Calculate weights based on distance
                            weights = 1.0 / (distances + 1e-10)**2
                            weights = weights / np.sum(weights)
                            
                            # Get actual 3D points
                            nearest_points = basal_points[indices]
                            
                            # Interpolate position using weighted average
                            interpolated_point = np.sum(nearest_points * weights[:, np.newaxis], axis=0)
                            
                            # Store the interpolated point
                            grid_points[i, j] = interpolated_point - center
                    
                    logging.debug(f"Created grid points with shape: {grid_points.shape}")
                    
                    # Convert to list format expected by NURBS
                    grid_points_list = []
                    for i in range(num_v):
                        for j in range(num_u):
                            point = grid_points[i, j].tolist()
                            grid_points_list.append(point)
                    
                    logging.debug(f"Converted to list format, length: {len(grid_points_list)}")
                    return grid_points_list

                except Exception as e:
                    logging.error(f"Error in create_boundary_grid: {str(e)}\n{traceback.format_exc()}")
                    return None

            # Create initial grid with boundary connection
            logging.info("Creating boundary-aware grid...")
            grid_points = create_boundary_grid(basal_points, control_points_u, control_points_v)
            
            # Fit NURBS surface
            logging.info("Fitting NURBS surface...")
            try:
                surf = BSpline.Surface()
                surf.degree_u = min(degree_u, control_points_u - 1)
                surf.degree_v = min(degree_v, control_points_v - 1)
                surf.ctrlpts_size_u = control_points_u
                surf.ctrlpts_size_v = control_points_v
                
                # Set control points directly
                surf.ctrlpts = grid_points
                logging.debug(f"Control points set with size: {control_points_u}x{control_points_v}")
                
                # Generate knot vectors
                surf.knotvector_u = utilities.generate_knot_vector(surf.degree_u, surf.ctrlpts_size_u)
                surf.knotvector_v = utilities.generate_knot_vector(surf.degree_v, surf.ctrlpts_size_v)
                
            except Exception as e:
                logging.error(f"Error in NURBS fitting: {str(e)}")
                return None

            # Generate surface points
            logging.info("Generating surface points...")
            try:
                surf.delta = 0.02  # Finer sampling
                surf.evaluate()
                surface_points = np.array(surf.evalpts)
                
                # Transform points back to original coordinate system
                surface_points = surface_points + center
                
                return surface_points
                
            except Exception as e:
                logging.error(f"Error in surface generation: {str(e)}")
                return None

        except Exception as e:
            logging.error(f"Error in bottom face generation: {str(e)}\n{traceback.format_exc()}")
            return None

    def save_mesh(self, file_path: Union[str, Path]) -> None:
        """
        Save the reconstructed mesh to a file
        
        Args:
            file_path: Path where to save the mesh
        """
        try:
            file_path = Path(file_path)
            if not str(file_path).lower().endswith('.ply'):
                file_path = file_path.with_suffix('.ply')
            
            if self.reconstructed_mesh is None:
                raise ValueError("No mesh to save. Please reconstruct the mesh first.")
                
            o3d.io.write_triangle_mesh(str(file_path), self.reconstructed_mesh)
            
            # Clean up temporary mesh file if it exists
            if self.temp_mesh_path and os.path.exists(self.temp_mesh_path):
                os.unlink(self.temp_mesh_path)
                
            logging.info(f"Mesh saved to {file_path}")

            return str(file_path)
            
        except Exception as e:
            logging.error(f"Error saving mesh: {str(e)}")
            raise

    def _ensure_boundary_connection(self, surface_points: np.ndarray,
                                  basal_points: np.ndarray,
                                  connection_threshold: float = 0.1) -> np.ndarray:
        """
        Ensures that the surface points connect smoothly with the basal points.
        
        Args:
            surface_points: Generated surface points
            basal_points: Original basal points
            connection_threshold: Maximum distance for connecting points
            
        Returns:
            np.ndarray: Modified surface points with boundary connection
        """
        logging.debug(f"Ensuring boundary connection with {len(basal_points)} basal points")
        
        try:
            # Build KD-tree for surface points
            surface_tree = cKDTree(surface_points)
            
            # Find nearest surface points for each basal point
            distances, indices = surface_tree.query(basal_points, k=5)  # Increased k for smoother transition
            
            # Create transition points
            transition_points = []
            for i, basal_point in enumerate(basal_points):
                # Get nearest surface points
                nearest_points = surface_points[indices[i]]
                weights = 1.0 / (distances[i] + 1e-6)
                weights = weights / np.sum(weights)
                
                # Create gradient of points from basal to surface
                for t in np.linspace(0, 1, 5):
                    # Weighted interpolation
                    weighted_surface_point = np.sum(nearest_points * weights[:, np.newaxis], axis=0)
                    interpolated = basal_point * (1 - t) + weighted_surface_point * t
                    transition_points.append(interpolated)

            # Combine all points
            if transition_points:
                surface_points = np.vstack((surface_points, transition_points, basal_points))
            else:
                surface_points = np.vstack((surface_points, basal_points))

            return surface_points

        except Exception as e:
            logging.error(f"Error in boundary connection: {str(e)}")
            return np.vstack((surface_points, basal_points))
