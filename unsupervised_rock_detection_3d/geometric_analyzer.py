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
                                  pedestal_points: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Compute geometric properties of a rock mesh.
        
        Args:
            mesh: Open3D triangle mesh of the rock
            basal_points: Array of basal point coordinates
            pedestal_points: Optional array of pedestal point coordinates
            
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

            # PCA for major orientations
            mesh_points = np.asarray(mesh.vertices)
            covariance_matrix = np.cov(mesh_points.T)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

            # Sort by eigenvalues in descending order
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Calculate dimensions along major axes
            points_transformed = np.dot(mesh_points - center_of_mass, eigenvectors)
            min_bounds = np.min(points_transformed, axis=0)
            max_bounds = np.max(points_transformed, axis=0)
            dimensions = max_bounds - min_bounds

            # Sort dimensions (length > width)
            length = max(dimensions[0], dimensions[1])
            width = min(dimensions[0], dimensions[1])
            height = dimensions[2]

            # Calculate ratios and face directions
            height_width_ratio = height / width
            length_width_ratio = length / width
            height_width_face = eigenvectors[:, 2]  # Normal to height-width plane
            length_width_face = eigenvectors[:, 0]  # Normal to length-width plane

            # Calculate alpha angle
            basal_com_vectors = basal_points - center_of_mass
            z_axis = np.array([0, 0, 1])
            angles = []
            for vector in basal_com_vectors:
                angle = np.arccos(np.dot(vector, z_axis) /
                                  (np.linalg.norm(vector) * np.linalg.norm(z_axis)))
                angles.append(np.degrees(angle))
            alpha_angle = min(angles)

            # Calculate beta angle (pedestal plane to vertical)
            # Fit plane to pedestal points
            pedestal_mean = np.mean(pedestal_points, axis=0)
            pedestal_covariance = np.cov(pedestal_points.T)
            _, pedestal_eigenvectors = np.linalg.eigh(pedestal_covariance)
            pedestal_normal = pedestal_eigenvectors[:, 0]
            beta_angle = np.degrees(np.arccos(np.abs(np.dot(pedestal_normal, z_axis))))

            return {
                'center_of_mass': center_of_mass,
                'major_orientations': eigenvectors,
                'height': height,
                'width': width,
                'length': length,
                'height_width_ratio': height_width_ratio,
                'height_width_face': height_width_face,
                'length_width_ratio': length_width_ratio,
                'length_width_face': length_width_face,
                'alpha_angle': alpha_angle,
                'beta_angle': beta_angle
            }

        except Exception as e:
            logging.error(f"Error computing geometric properties: {str(e)}")
            raise

    def save_results(self, results: Dict[str, Any], 
                    pbr_name: str, 
                    input_path: Union[str, Path], 
                    segmented_path: Union[str, Path], 
                    mesh_path: Union[str, Path],
                    output_csv: Optional[Union[str, Path]] = None) -> None:
        """
        Save analysis results to a CSV file.
        
        Args:
            results: Dictionary containing analysis results
            pbr_name: Name of the PBR being analyzed
            input_path: Path to input point cloud
            segmented_path: Path to segmented point cloud
            mesh_path: Path to reconstructed mesh
            output_csv: Optional path to output CSV file
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
                'beta_angle': results['beta_angle']
            }
            
            csv_path = input_path.parent / f"{str(input_path.parent).split('/')[-1]}_geometric_analysis_results.csv"
            if not csv_path.exists():
                pd.DataFrame([data]).to_csv(csv_path, index=False)
            else:
                pd.DataFrame([data]).to_csv(csv_path, mode='a', header=False, index=False)

            return str(csv_path)
                
        except Exception as e:
            logging.error(f"Error saving results: {str(e)}")
            raise