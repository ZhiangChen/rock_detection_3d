import numpy as np
import laspy
import open3d as o3d
from pathlib import Path
import csv
import logging
from scipy.spatial import ConvexHull
import pandas as pd
from scipy.spatial import cKDTree
from geomdl import BSpline
from geomdl import utilities
import logging
import traceback
import csv

class PBRGeometricAnalyzer:
    def __init__(self):
        self.x_mean = 0
        self.y_mean = 0
        self.z_mean = 0
        
    def load_segmented_las(self, las_file_path):
        """Load segmented LAS file with labels in Intensity field"""
        try:
            pc = laspy.read(las_file_path)
            x, y, z = pc.x, pc.y, pc.z
            labels = pc.intensity  # Labels stored in intensity field
            
            # Store mean values for recentering
            self.x_mean = np.mean(x)
            self.y_mean = np.mean(y)
            self.z_mean = np.mean(z)
            
            # Recenter points
            xyz = np.vstack((x - self.x_mean, y - self.y_mean, z - self.z_mean)).transpose()
            
            # Create Open3D PointCloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            
            # Separate points by labels
            rock_mask = labels == 1
            basal_mask = labels == 2
            pedestal_mask = labels == 0
            
            return pcd, xyz, rock_mask, basal_mask, pedestal_mask
            
        except Exception as e:
            logging.error(f"Error loading LAS file: {str(e)}")
            raise

    def generate_bottom_face_points(self, pcd, basal_indices, degree_u=3, degree_v=3, 
                                  control_points_u=10, control_points_v=10):
        """Generate bottom face points with support for multiple parts"""
        try:
            logging.info("Starting bottom face generation")
            
            # Check if we're dealing with multiple parts
            if hasattr(self, 'basal_parts') and len(self.basal_parts) > 1:
                logging.info(f"Generating faces for {len(self.basal_parts)} parts")
                return self._generate_multi_part_faces(pcd, self.basal_parts, 
                                                     degree_u, degree_v,
                                                     control_points_u, control_points_v)
            else:
                # Single part processing
                logging.info("Generating single face")
                return self._generate_single_face(pcd, basal_indices, 
                                                degree_u, degree_v,
                                                control_points_u, control_points_v,
                                                is_dense_points=False)
                
        except Exception as e:
            logging.error(f"Error in bottom face generation: {str(e)}")
            return None
        
    def _generate_single_face(self, pcd, points_or_indices, degree_u, degree_v, 
                            control_points_u, control_points_v, is_dense_points=False):
        """Original bottom face generation logic for a single part"""
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
                surface_points = self.ensure_boundary_connection(surface_points, basal_points)
                
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
            logging.error(f"Error in bottom face generation: {str(e)}")
            return None
        
    def ensure_boundary_connection(self, surface_points, basal_points, connection_threshold=0.1):
        """
        Ensures that the surface points connect smoothly with the basal points.
        
        Args:
            surface_points (np.ndarray): Generated surface points
            basal_points (np.ndarray): Original basal points
            connection_threshold (float): Maximum distance for connecting points
            
        Returns:
            np.ndarray: Modified surface points with boundary connection
        """
        logging.debug(f"Surface points shape: {surface_points.shape}")
        logging.debug(f"Basal points shape: {basal_points.shape}")
        try:
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

    def reconstruct_mesh(self, pcd, rock_points, basal_points):
        """Reconstruct watertight mesh from point cloud using NURBS and Poisson reconstruction"""
        try:
            # Combine rock and basal points
            combined_points = np.vstack((rock_points, basal_points))
            mesh_pcd = o3d.geometry.PointCloud()
            mesh_pcd.points = o3d.utility.Vector3dVector(combined_points)
            
            # Estimate normals for the combined point cloud
            mesh_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            
            # Generate bottom face points using NURBS interpolation
            bottom_points = self.generate_bottom_face_points(
                mesh_pcd,
                np.arange(len(basal_points)),  # Assuming basal_points are at the end
                degree_u=3,
                degree_v=3,
                control_points_u=12,
                control_points_v=12
            )
            
            if bottom_points is None:
                logging.error("Failed to generate bottom face points")
                raise ValueError("Failed to generate bottom face points")
            
            # Combine rock points and bottom face points
            combined_points = np.vstack((rock_points, bottom_points))
            combined_normals = np.vstack((
                np.asarray(mesh_pcd.normals),
                np.zeros_like(bottom_points)  # Initialize normals for bottom points
            ))
            
            # Create final point cloud
            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(combined_points)
            new_pcd.normals = o3d.utility.Vector3dVector(combined_normals)
            
            # Estimate normals for the new point cloud
            new_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            
            # Orient normals
            center = new_pcd.get_center()
            new_pcd.orient_normals_towards_camera_location(center)
            new_pcd.normals = o3d.utility.Vector3dVector(-np.asarray(new_pcd.normals))
            
            # Poisson reconstruction
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                new_pcd, depth=8, linear_fit=False
            )
            
            # Convert to tensor-based TriangleMesh
            tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            
            # Fill holes
            filled_mesh = tensor_mesh.fill_holes()
            
            # Convert back to legacy mesh if needed
            filled_mesh_legacy = filled_mesh.to_legacy()
            
            return filled_mesh_legacy
            
        except Exception as e:
            logging.error(f"Error in mesh reconstruction: {str(e)}")
            raise

    def compute_geometric_properties(self, mesh, xyz, rock_mask, basal_mask, pedestal_mask):
        """Compute geometric properties of the PBR"""
        try:
            # # Ensure the mesh is triangulated
            # if not mesh.is_triangle_mesh():
            #     mesh = mesh.triangulate()

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

            # Get points for each component
            rock_points = xyz[rock_mask]
            basal_points = xyz[basal_mask]
            pedestal_points = xyz[pedestal_mask]

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

    def save_results(self, results, pbr_name, input_path, output_path, mesh_path):
        """Save results to CSV file"""
        try:
            data = {
                'pbr_name': pbr_name,
                'pbr_location': str(input_path),
                'segmented_pbr_location': str(output_path),
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
            
            csv_path = output_path.parent / 'geometric_analysis_results.csv'
            if not csv_path.exists():
                pd.DataFrame([data]).to_csv(csv_path, index=False)
            else:
                pd.DataFrame([data]).to_csv(csv_path, mode='a', header=False, index=False)
                
        except Exception as e:
            logging.error(f"Error saving results: {str(e)}")
            raise

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    analyzer = PBRGeometricAnalyzer()
    
    # Example usage
    input_path = Path("/Users/deeprodge/Downloads/DREAMS/PG&E/rock_detection_3d/unsupervised_rock_detection_3d/PBR FP 10_segmented.las")
    output_path = input_path.parent / f"{input_path.stem}_analyzed{input_path.suffix}"
    mesh_path = input_path.parent / f"{input_path.stem}_mesh.ply"
    
    try:
        # Load and process point cloud
        pcd, xyz, rock_mask, basal_mask, pedestal_mask = analyzer.load_segmented_las(input_path)
        
        # Reconstruct mesh
        mesh = analyzer.reconstruct_mesh(pcd, xyz[rock_mask], xyz[basal_mask])
        
        # Save mesh
        o3d.io.write_triangle_mesh(str(mesh_path), mesh)
        
        # Compute geometric properties
        results = analyzer.compute_geometric_properties(
            mesh, xyz, rock_mask, basal_mask, pedestal_mask
        )
        
        # Save results
        analyzer.save_results(
            results,
            input_path.stem,
            input_path,
            output_path,
            mesh_path
        )
        
        logging.info("Geometric analysis completed successfully")
        
    except Exception as e:
        logging.error(f"Error in geometric analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 