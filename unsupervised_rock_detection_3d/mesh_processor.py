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

class MeshProcessor:
    """
    Handles all mesh-related operations including mesh reconstruction,
    bottom face generation, and mesh saving.
    """
    
    def __init__(self):
        self.temp_mesh_path = None
        self.reconstructed_mesh = None

    def reconstruct_mesh(self, pcd: o3d.geometry.PointCloud, labels: np.ndarray, 
                        basal_points: np.ndarray, dense_basal_parts: list = None, degree_u: int = 3, degree_v: int = 3,
                        control_points_u: int = 12, control_points_v: int = 12) -> o3d.geometry.TriangleMesh:
        """
        Reconstructs a 3D mesh from the segmented point cloud.
        Process:
        1. Filters points to keep rock and basal points
        2. Generates bottom face using NURBS interpolation
        3. Performs Poisson reconstruction
        
        Args:
            pcd: Open3D PointCloud object
            labels: Array of point labels (0 for pedestal, 1 for rock)
            basal_points: Array of basal point indices or boolean mask
            degree_u, degree_v: Degrees for NURBS surface
            control_points_u, control_points_v: Number of control points for NURBS surface
            
        Returns:
            o3d.geometry.TriangleMesh: Reconstructed mesh
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

            # Get indices for basal points in filtered point cloud
            basal_indices = np.where(basal_mask[filtered_indices])[0]
            rock_indices = np.where(rock_points[filtered_indices])[0]

            # Generate bottom face points
            bottom_points = self.generate_bottom_face_points(
                filtered_pcd,
                basal_indices,
                dense_basal_parts=dense_basal_parts,
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
            rock_pcd.orient_normals_consistent_tangent_plane(100)
            
            # Process bottom points
            bottom_pcd = o3d.geometry.PointCloud()
            bottom_pcd.points = o3d.utility.Vector3dVector(bottom_points)
            bottom_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            bottom_pcd.orient_normals_consistent_tangent_plane(100)
            
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
            
            # Orient normals consistently
            new_pcd.orient_normals_consistent_tangent_plane(100)
            
            # Ensure normals point outward
            center = new_pcd.get_center()
            new_pcd.orient_normals_towards_camera_location(center)
            new_pcd.normals = o3d.utility.Vector3dVector(-np.asarray(new_pcd.normals))
            
            # Color the points
            new_pcd.paint_uniform_color([1, 0, 0])
            # Reconstruct mesh with improved parameters
            self.reconstructed_mesh = self.poisson_reconstruction(
                new_pcd,
                depth=9,        # Increased depth for better detail
                width=0.05,     # Added density filtering
                scale=1.1,      # Slightly larger scale
                linear_fit=True # Enable linear fit for better results
            )
            
            # Save temporary mesh for visualization
            with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as temp_file:
                self.temp_mesh_path = temp_file.name
            o3d.io.write_triangle_mesh(self.temp_mesh_path, self.reconstructed_mesh)

            return self.reconstructed_mesh

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
        Performs Poisson surface reconstruction on the point cloud with improved robustness.
        
        Args:
            pcd: Open3D PointCloud object
            depth: Octree depth, controls the resolution. Higher values give finer details but more noise
            width: Width parameter for density filtering
            scale: Scale factor for reconstruction
            linear_fit: Whether to use linear fit
            
        Returns:
            o3d.geometry.TriangleMesh: Reconstructed mesh
        """
        try:
            # cleaned_pcd, _ = pcd.remove_statistical_outlier(
            #     nb_neighbors=20,
            #     std_ratio=2.0
            # )
            
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
               pcd, depth=8, linear_fit=False)

            
            # # Clean up the mesh
            # mesh.remove_degenerate_triangles()
            # mesh.remove_duplicated_triangles()
            # mesh.remove_duplicated_vertices()
            # mesh.remove_non_manifold_edges()
            return mesh

        except Exception as e:
            logging.error(f"Error in Poisson reconstruction: {str(e)}")
            raise

    def generate_bottom_face_points(self, pcd: o3d.geometry.PointCloud, 
                                  basal_indices: np.ndarray,
                                  dense_basal_parts: list = None,
                                  degree_u: int = 3, 
                                  degree_v: int = 3,
                                  control_points_u: int = 10,
                                  control_points_v: int = 10 ) -> np.ndarray:
        """
        Generate bottom face points with support for multiple parts
        
        Args:
            pcd: Open3D PointCloud object
            basal_indices: Array of indices for basal points
            degree_u, degree_v: Degrees for NURBS surface
            control_points_u, control_points_v: Number of control points
            multi_part_data: Dictionary containing data for multiple parts
            
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
                                 degree_u: int,
                                 degree_v: int,
                                 control_points_u: int,
                                 control_points_v: int) -> np.ndarray:
        """
        Generate separate faces for each basal part using dense basal points
        
        Args:
            pcd: Open3D PointCloud object
            dense_basal_parts: List of dense basal points for each part
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
                logging.info(f"Processing dense part {i+1}/{len(dense_basal_parts)}")
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
                
                # Ensure connection with basal points
                surface_points = self._ensure_boundary_connection(surface_points, basal_points)
                
                # Add additional points near the boundary
                tree = cKDTree(basal_points)
                boundary_points = []
                
                for basal_point in basal_points:
                    # Add points slightly inward from each basal point
                    inward_vector = center - basal_point
                    inward_vector = inward_vector / np.linalg.norm(inward_vector)
                    
                    # Add multiple points at different distances
                    for d in [0.02, 0.1]:  # Adjust these distances as needed
                        new_point = basal_point + inward_vector * d
                        boundary_points.append(new_point)
                
                # Combine all points
                all_points = np.vstack((surface_points, boundary_points))
                
                logging.info(f"Generated {len(all_points)} total points for the bottom face")
                return all_points
                
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

    @staticmethod
    def _ensure_boundary_connection(surface_points: np.ndarray,
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
            distances, indices = surface_tree.query(basal_points, k=1)
            
            # Identify gaps where surface points are too far from basal points
            gaps = distances > connection_threshold
            
            if np.any(gaps):
                logging.debug(f"Found {np.sum(gaps)} gaps to fill")
                gap_points = []
                
                for basal_point, surface_idx, has_gap in zip(basal_points, indices, gaps):
                    if has_gap:
                        surface_point = surface_points[surface_idx]
                        
                        # Create intermediate points to fill the gap
                        num_intermediate = 3  # Number of intermediate points
                        for t in np.linspace(0, 1, num_intermediate + 2)[1:-1]:
                            intermediate_point = basal_point * (1 - t) + surface_point * t
                            gap_points.append(intermediate_point)
                
                # Add gap-filling points to surface points
                if gap_points:
                    surface_points = np.vstack((surface_points, gap_points))
                    logging.debug(f"Added {len(gap_points)} connection points")
            
            # Ensure all basal points are included
            surface_points = np.vstack((surface_points, basal_points))
            logging.debug("Successfully connected surface to basal points")
            
            return surface_points
        except Exception as e:
            logging.error(f"Error in boundary connection: {str(e)}\n{traceback.format_exc()}")
            # If there's an error, return the original surface points
            return surface_points