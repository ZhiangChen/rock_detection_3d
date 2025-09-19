import open3d as o3d
import numpy as np
import logging
from pathlib import Path
import csv
from typing import Dict, Any, Optional, Union
import pandas as pd

class GeometricAnalyzer:
    """
    Handles geometric analysis operations for 3D point clouds and meshes.
    """

    def compute_geometric_properties(self, mesh: o3d.geometry.TriangleMesh, 
                                  basal_points: np.ndarray,
                                  pedestal_points: Optional[np.ndarray] = None,
                                  lateral_flags: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Compute geometric properties of a rock mesh.
        
        Args:
            mesh: Open3D triangle mesh of the rock
            basal_points: Array of basal point coordinates
            pedestal_points: Optional array of pedestal point coordinates
            lateral_flags: Optional array of boolean flags indicating which basal points are lateral
            
        Returns:
            dict: Dictionary containing computed geometric properties
        """
        try:


            # Initialize accumulators for volume and weighted centroids
            total_volume = 0.0
            weighted_centroid_sum = np.zeros(3)

            # Reference point (origin)
            reference_point = np.array([0.0, 0.0, 0.0])

            # Iterate over triangles
            for triangle in mesh.triangles:
                vertices = np.asarray(mesh.vertices)[triangle]
                v0, v1, v2 = vertices

                # Calculate the signed volume of the tetrahedron
                tetra_volume = np.dot(np.cross(v0 - reference_point, v1 - reference_point), v2 - reference_point) / 6.0

                # Calculate the centroid of the tetrahedron
                tetra_centroid = (v0 + v1 + v2 + reference_point) / 4.0

                # Accumulate the volumes and weighted centroids
                total_volume += tetra_volume
                weighted_centroid_sum += tetra_volume * tetra_centroid

            # Compute the final center of mass
            center_of_mass = weighted_centroid_sum / total_volume

            # Compute the mesh centroid (average of all vertices)
            mesh_points = np.asarray(mesh.vertices)
            centroid = np.mean(mesh_points, axis=0)
            logging.debug(f"Mesh centroid: {centroid}")
            logging.debug(f"Center of mass: {center_of_mass}")

            # PCA for major orientations
            mesh_points = np.asarray(mesh.vertices)
            centroid = np.mean(mesh_points, axis=0)
            
            # Center the points by subtracting centroid
            centered_points = mesh_points - centroid

            # Compute PCA
            covariance_matrix = np.cov(centered_points.T)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

            # Sort by eigenvalues in descending order
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Project points onto PCA axes
            points_in_pca = np.dot(centered_points, eigenvectors)
            
            # Calculate dimensions in PCA space
            min_bounds = np.min(points_in_pca, axis=0)
            max_bounds = np.max(points_in_pca, axis=0)
            dimensions = max_bounds - min_bounds

            # Assign dimensions:
            # - height is aligned with Y (second PC)
            # - width is the smaller of remaining dimensions (X and Z)
            # - length is the larger of remaining dimensions
            height = dimensions[1]  # Y-axis aligned dimension (second PC)
            
            # Find width (smaller) and length (larger) from remaining dimensions
            other_dims = np.array([dimensions[0], dimensions[2]])  # X and Z dimensions
            
            # Get indices for width and length from the remaining dimensions
            width_idx = 0 if dimensions[0] < dimensions[2] else 2
            length_idx = 2 if dimensions[0] < dimensions[2] else 0
            
            width = dimensions[width_idx]
            length = dimensions[length_idx]
            
            # Calculate ratios
            height_width_ratio = height / width
            length_width_ratio = length / width
            
            # Face directions correspond to eigenvectors
            # height_width_face should be normal to the plane containing height (Y) and width
            # length_width_face should be normal to the plane containing length and width
            height_width_face = eigenvectors[:, length_idx]  # Normal to height-width plane
            length_width_face = eigenvectors[:, 1]  # Normal to length-width plane (Y axis)

            logging.debug(f"Face normals - Height-Width: {height_width_face}, Length-Width: {length_width_face}")

            logging.debug(f"PCA eigenvalues: {eigenvalues}")
            logging.debug(f"Raw dimensions in PCA space: {dimensions}")
            logging.debug(f"PCA dimensions - X: {dimensions[0]:.3f}, Y (height): {dimensions[1]:.3f}, Z: {dimensions[2]:.3f}")
            logging.debug(f"Assigned dimensions - Height (Y): {height:.3f}, Width (min X/Z): {width:.3f}, Length (max X/Z): {length:.3f}")

            # For verification, add the ranges in original space
            original_ranges = np.max(mesh_points, axis=0) - np.min(mesh_points, axis=0)
            logging.debug(f"Original space ranges: {original_ranges}")

            # Calculate ratios using MeshLab's ordering
            height_width_ratio = height / width
            length_width_ratio = length / width
            
            # Face directions still correspond to eigenvectors
            height_width_face = eigenvectors[:, 2]  # Normal to height-width plane
            length_width_face = eigenvectors[:, 0]  # Normal to length-width plane

            # Calculate alpha angle - CORRECTED IMPLEMENTATION
            # Alpha angle is the angle between CoM-to-basal-point vector and negative Z axis
            # Only use non-lateral basal points for this calculation
            neg_z_axis = np.array([0, 0, -1])  # Downward Z vector
            angles = []
            min_alpha_basal_point = None
            min_angle = float('inf')
            
            # Filter basal points to exclude lateral points
            if lateral_flags is not None:
                # Only use points that are not lateral (False in lateral_flags)
                non_lateral_mask = ~lateral_flags
                alpha_basal_points = basal_points[non_lateral_mask]
                logging.info(f"Using {len(alpha_basal_points)} non-lateral points out of {len(basal_points)} total basal points for alpha angle calculation")
                
                # Log the distribution of lateral vs non-lateral points
                num_lateral = np.sum(lateral_flags)
                num_non_lateral = len(lateral_flags) - num_lateral
                logging.info(f"Point distribution: {num_lateral} lateral, {num_non_lateral} non-lateral")
                
            else:
                # Use all basal points if no lateral flags provided
                alpha_basal_points = basal_points
                logging.info(f"Using all {len(alpha_basal_points)} basal points for alpha angle calculation (no lateral flags provided)")
            
            for basal_point in alpha_basal_points:
                # Vector FROM center of mass TO basal point (downward)
                vector = basal_point - center_of_mass
                vector_norm = np.linalg.norm(vector)
                if vector_norm < 1e-10:  # Skip if vector is too small
                    continue
                unit_vector = vector / vector_norm
                
                # Calculate angle using dot product of downward vectors
                cos_angle = np.dot(unit_vector, neg_z_axis)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                angle_degrees = np.degrees(angle)
                angles.append(angle_degrees)
                
                # Track the basal point with minimum angle
                if angle < min_angle:
                    min_angle = angle
                    min_alpha_basal_point = basal_point
            
            if not angles:
                alpha_angle = 0.0  # Default if no valid angles found
                min_alpha_basal_point = center_of_mass  # Default to center of mass
                logging.warning("No valid angles found for alpha calculation")
            else:
                alpha_angle = min(angles)
                logging.info(f"Alpha angle calculation: minimum angle = {alpha_angle:.2f} degrees from {len(angles)} valid angles")

            # Calculate the alpha plane normal vector
            # The plane consists of: global Z axis, center of mass, and min alpha basal point
            z_axis = np.array([0, 0, 1])  # Global Z axis (upward)
            com_to_basal = min_alpha_basal_point - center_of_mass
            
            # Calculate plane normal using cross product
            # Normal to plane containing Z axis and CoM-to-basal vector
            alpha_plane_normal = np.cross(z_axis, com_to_basal)
            alpha_plane_normal_norm = np.linalg.norm(alpha_plane_normal)
            
            if alpha_plane_normal_norm > 1e-10:
                alpha_plane_normal = alpha_plane_normal / alpha_plane_normal_norm
            else:
                # If vectors are parallel, use a default normal
                alpha_plane_normal = np.array([1, 0, 0])  # Default to X axis
            
            # Calculate alpha angle assuming a rectangular cross-section
            # Use half of the height and half of the width
            alpha_rectangular = np.degrees(np.arctan((width/ height)))

            # Calculate beta angle (pedestal plane to vertical)
            # Fit plane to pedestal points
            pedestal_mean = np.mean(pedestal_points, axis=0)
            pedestal_covariance = np.cov(pedestal_points.T)
            _, pedestal_eigenvectors = np.linalg.eigh(pedestal_covariance)
            pedestal_normal = pedestal_eigenvectors[:, 0]
            z_axis = np.array([0, 0, -1])
            beta_angle = np.degrees(np.arccos(np.abs(np.dot(pedestal_normal, z_axis))))

            return {
                'center_of_mass': center_of_mass,
                'height': height,
                'width': width, 
                'length': length,
                'major_orientations': eigenvectors,
                'height_width_ratio': height_width_ratio,
                'height_width_face': height_width_face,
                'length_width_ratio': length_width_ratio,
                'length_width_face': length_width_face,
                'alpha_angle': alpha_angle,
                'alpha_rectangular': alpha_rectangular,
                'beta_angle': beta_angle,
                'min_alpha_basal_point': min_alpha_basal_point,  # Add the basal point with min angle
                'alpha_plane_normal': alpha_plane_normal,  # Add the alpha plane normal
            }

        except Exception as e:
            logging.error(f"Error computing geometric properties: {str(e)}")
            raise

    def save_results(self, results: Dict[str, Any], 
                    pbr_name: str, 
                    input_path: Union[str, Path], 
                    segmented_path: Union[str, Path], 
                    mesh_path: Union[str, Path],
                    smoothness_threshold: Optional[float] = None,
                    curvature_threshold: Optional[float] = None,
                    proximity_threshold: Optional[float] = None,
                    user: Optional[str] = None,
                    epsg_code: Optional[int] = None,
                    output_csv: Optional[Union[str, Path]] = None) -> None:
        """
        Save analysis results to a CSV file.
        
        Args:
            results: Dictionary containing analysis results
            pbr_name: Name of the PBR being analyzed
            input_path: Path to input point cloud
            segmented_path: Path to segmented point cloud
            mesh_path: Path to reconstructed mesh
            smoothness_threshold: Optional threshold used for smoothness analysis
            curvature_threshold: Optional threshold used for curvature analysis
            proximity_threshold: Optional threshold used for proximity analysis
            user: Name of the user performing the analysis
            epsg_code: Optional EPSG code for the coordinate system
            output_csv: Optional path to output CSV file (if provided, results will be appended to this file)
        """
        try:
            data = {
                'pbr_name': pbr_name,
                'pbr_location': str(input_path),
                'segmented_pbr_location': str(segmented_path),
                'mesh_reconstruction_location': str(mesh_path),
                'height': results['height'],
                'width': results['width'],
                'length': results['length'],
                'center_of_mass': results['center_of_mass'].tolist(),
                'major_orientations': results['major_orientations'].tolist(),
                'height_width_ratio': results['height_width_ratio'],
                'height_width_face': results['height_width_face'].tolist(),
                'length_width_ratio': results['length_width_ratio'],
                'length_width_face': results['length_width_face'].tolist(),
                'alpha_angle': results['alpha_angle'],
                'alpha_rectangular': results['alpha_rectangular'],  # Add alpha angle for rectangular cross-section
                'beta_angle': results['beta_angle'],
                'smoothness_threshold': smoothness_threshold,
                'curvature_threshold': curvature_threshold,
                'proximity_threshold': proximity_threshold,
                'epsg_code': epsg_code,  # Add EPSG code to data dictionary
                'user': user,  # Add user to the data dictionary
            }
            
            # Use provided CSV path if available, otherwise use default path
            if output_csv is not None:
                csv_path = Path(output_csv)
            else:
                csv_path = input_path.parent / f"{str(input_path.parent).split('/')[-1]}_geometric_analysis_results.csv"
            
            # Create DataFrame from the data
            df = pd.DataFrame([data])
            
            # Check if file exists to determine if we need headers
            file_exists = csv_path.exists()
            
            if file_exists:
                # Append to existing file without headers
                df.to_csv(csv_path, mode='a', header=False, index=False)
            else:
                # Create new file with headers
                df.to_csv(csv_path, mode='w', header=True, index=False)

            return str(csv_path)
                
        except Exception as e:
            logging.error(f"Error saving results: {str(e)}")
            raise