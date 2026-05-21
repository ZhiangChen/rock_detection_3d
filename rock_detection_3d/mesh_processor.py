import open3d as o3d
import numpy as np
import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass
from geomdl import BSpline
from geomdl import utilities
from scipy.spatial import cKDTree
import numpy as np
import open3d as o3d
import logging
from sklearn.cluster import DBSCAN
from multiprocessing.connection import Connection
import traceback
import os
from sklearn.cluster import DBSCAN
from multiprocessing.connection import Connection

# Try to import PyMeshLab
try:
    import pymeshlab
    PYMESHLAB_AVAILABLE = True
    logging.info("PyMeshLab is available for enhanced normal computation")
except ImportError:
    PYMESHLAB_AVAILABLE = False
    logging.warning("PyMeshLab not available. Using fallback normal computation methods.")

from visualization import PointCloudVisualization

@dataclass
class BottomFacePreparationResult:
    """Clean data structure for bottom face preparation results"""
    rock_points: np.ndarray
    bottom_points: np.ndarray
    basal_indices: np.ndarray

class MeshProcessor:
    """
    Handles all mesh-related operations including mesh reconstruction,
    bottom face generation, and mesh saving.
    """
    
    def __init__(self, noise_settings: Optional[dict] = None):
        self.temp_mesh_path = None
        self.reconstructed_mesh = None
        self.last_error_message: str = ""
        default_noise_settings = {
            "sor_neighbors": 100,
            "sor_std_ratio": 2.0,
            "cluster_cleanup": True,
            "cluster_eps": 0.02,  # Fixed at 2cm for stability
            "cluster_dbscan_min_points": 10,
            "cluster_min_pct": 0.01,  # 1% of total points
            "basal_clipping": True,  # Enable clipping against basal surface
            "basal_clip_threshold": 0.0,  # Exact surface by default
        }
        self.noise_settings = {**default_noise_settings, **(noise_settings or {})}

    def clean_outliers_dbscan(self, pcd: o3d.geometry.PointCloud, 
                             eps: float = 0.02, 
                             min_samples: int = 10,
                             return_inlier_indices: bool = False) -> Union[o3d.geometry.PointCloud, Tuple[o3d.geometry.PointCloud, np.ndarray]]:
        """
        Remove outliers using DBSCAN clustering algorithm with improved multi-cluster preservation.
        
        Args:
            pcd: Input point cloud
            eps: Maximum distance between points in the same cluster (default 0.02m = 2cm)
            min_samples: Minimum number of points to form a dense region (default 10)
            return_inlier_indices: Whether to return indices of inlier points
            
        Returns:
            Cleaned point cloud and optionally indices of inlier points
        """
        try:
            points = np.asarray(pcd.points)
            initial_count = len(points)
            
            if initial_count == 0:
                logging.warning("Empty point cloud provided to DBSCAN")
                if return_inlier_indices:
                    return pcd, np.array([], dtype=int)
                return pcd
            
            # Run DBSCAN clustering
            labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_samples, print_progress=False))
            
            # Count clusters (excluding noise labeled as -1)
            max_label = labels.max()
            if max_label < 0:
                logging.warning("No clusters found by DBSCAN, keeping all points")
                if return_inlier_indices:
                    return pcd, np.arange(initial_count)
                return pcd
            
            # Count points in each cluster
            cluster_counts = np.bincount(labels[labels >= 0])
            
            # Get cluster_min_pct from settings, default to 1%
            settings = getattr(self, "noise_settings", {})
            cluster_min_pct = float(settings.get("cluster_min_pct", 0.01))
            min_cluster_size = int(cluster_min_pct * initial_count)
            
            # Keep all clusters above the threshold (preserves split rock surfaces)
            valid_clusters = np.where(cluster_counts >= min_cluster_size)[0]
            
            logging.info(f"DBSCAN found {max_label + 1} clusters")
            logging.info(f"Minimum cluster size: {min_cluster_size} points (1% of {initial_count})")
            logging.info(f"Keeping {len(valid_clusters)} clusters above threshold")
            
            # Create mask for all valid clusters
            keep_mask = np.isin(labels, valid_clusters)
            inlier_indices = np.where(keep_mask)[0]
            
            cleaned_pcd = pcd.select_by_index(inlier_indices)
            removed_count = initial_count - len(inlier_indices)
            
            logging.info(f"Removed {removed_count} points ({100*removed_count/initial_count:.1f}%) via DBSCAN")
            
            if return_inlier_indices:
                return cleaned_pcd, inlier_indices
            return cleaned_pcd
            
        except Exception as e:
            logging.error(f"Error in DBSCAN clustering: {e}")
            if return_inlier_indices:
                return pcd, np.arange(len(np.asarray(pcd.points)))
            return pcd

    def clip_points_against_basal_surface(self, rock_points: np.ndarray, bottom_points: np.ndarray,
                                         trim_threshold: float = 0.0) -> Tuple[np.ndarray, dict]:
        """
        Clip rock points that extend below the basal surface using signed distance.
        
        Args:
            rock_points: Array of rock point coordinates (N, 3)
            bottom_points: Array of bottom face point coordinates (M, 3)
            trim_threshold: Distance threshold in meters (points with d >= -trim_threshold are kept)
                          0.0 = exact surface, 0.02 = allow 2cm penetration
            
        Returns:
            tuple: (clipped_rock_points, stats_dict)
        """
        try:
            if len(rock_points) == 0 or len(bottom_points) == 0:
                logging.warning("Empty points provided to clipping, skipping")
                return rock_points, {'kept_points': len(rock_points), 'removed_points': 0}
            
            # Compute surface normal using PCA
            centered = bottom_points - np.mean(bottom_points, axis=0)
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            normal = eigenvectors[:, 0]  # Normal is eigenvector with smallest eigenvalue
            
            # Ensure normal points upward (positive Z component)
            if normal[2] < 0:
                normal = -normal
            
            # Build KD-tree for bottom points
            kdtree = cKDTree(bottom_points)
            
            # For each rock point, find closest basal point
            distances, closest_indices = kdtree.query(rock_points, k=1)
            
            # Compute signed distances
            vectors = rock_points - bottom_points[closest_indices]
            signed_distances = np.dot(vectors, normal)
            
            # Apply threshold - keep points with signed_distance >= -trim_threshold
            keep_mask = signed_distances >= -trim_threshold
            
            clipped_rock_points = rock_points[keep_mask]
            removed_count = len(rock_points) - len(clipped_rock_points)
            
            stats = {
                'total_rock_points': len(rock_points),
                'kept_points': len(clipped_rock_points),
                'removed_points': removed_count,
                'removal_percentage': 100 * removed_count / len(rock_points) if len(rock_points) > 0 else 0,
                'min_signed_distance': np.min(signed_distances),
                'max_signed_distance': np.max(signed_distances),
                'below_surface_count': np.sum(signed_distances < 0),
            }
            
            logging.info(f"Basal clipping (threshold={trim_threshold:.3f}m):")
            logging.info(f"  Kept: {stats['kept_points']} points ({100 - stats['removal_percentage']:.1f}%)")
            logging.info(f"  Removed: {stats['removed_points']} points ({stats['removal_percentage']:.1f}%)")
            logging.info(f"  Points below surface: {stats['below_surface_count']}")
            
            return clipped_rock_points, stats
            
        except Exception as e:
            logging.error(f"Error in basal clipping: {e}")
            return rock_points, {'kept_points': len(rock_points), 'removed_points': 0}
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

    def prepare_bottom_face(self, pcd: o3d.geometry.PointCloud, labels: np.ndarray, 
                           basal_points: np.ndarray, dense_basal_parts: list = None, 
                           dense_basal_parts_is_lateral: list = None, degree_u: int = 4, degree_v: int = 4,
                           control_points_u: int = 5, control_points_v: int = 5, 
                           use_dbscan_cleaning: bool = False, 
                           basal_parts_metadata: dict = None) -> BottomFacePreparationResult:
        """
        Prepares rock and bottom face points for mesh reconstruction.
        This is the first stage of the pipeline - it filters points and generates the bottom face.
        
        Args:
            pcd: Open3D PointCloud object
            labels: Array of point labels (0 for pedestal, 1 for rock)
            basal_points: Array of basal point indices or boolean mask (legacy format)
            dense_basal_parts: List of dense basal parts (legacy format)
            dense_basal_parts_is_lateral: List of boolean flags indicating which parts are lateral (legacy format)
            degree_u, degree_v: Degrees for NURBS surface
            control_points_u, control_points_v: Number of control points for NURBS surface
            use_dbscan_cleaning: Whether to use DBSCAN for outlier removal
            basal_parts_metadata: Enhanced basal parts metadata structure (preferred format)
            
        Returns:
            BottomFacePreparationResult: Clean data structure containing rock points, bottom points, and basal indices
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

            # Extract dense basal parts from enhanced metadata if available
            if basal_parts_metadata is not None and 'parts' in basal_parts_metadata:
                # Extract from enhanced metadata
                dense_basal_parts = [
                    np.asarray(part.get('dense_points', []), dtype=float)
                    for part in basal_parts_metadata['parts']
                ]
                dense_basal_parts_is_lateral = [part.get('is_lateral', False) for part in basal_parts_metadata['parts']]
                logging.info(f"Using enhanced basal metadata with {len(dense_basal_parts)} parts")
            elif dense_basal_parts is None:
                # Fallback: no dense parts available
                dense_basal_parts = []
                dense_basal_parts_is_lateral = []
                logging.info("No dense basal parts available (neither enhanced metadata nor legacy format)")
            else:
                # Use legacy format
                logging.info(f"Using legacy basal parts format with {len(dense_basal_parts)} parts")

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
                control_points_v=control_points_v,
                basal_parts_metadata=basal_parts_metadata
            )
            
            if bottom_points is None:
                logging.error("Failed to generate bottom face points")
                raise ValueError("Failed to generate bottom face points")
            
            # Process rock points
            rock_points_array = np.asarray(filtered_pcd.points)[rock_indices]
            
            # Apply basal clipping if enabled
            if self.noise_settings.get('basal_clipping', False):
                clip_threshold = self.noise_settings.get('basal_clip_threshold', 0.0)
                rock_points_array, clip_stats = self.clip_points_against_basal_surface(
                    rock_points_array, bottom_points, trim_threshold=clip_threshold
                )
            
            return BottomFacePreparationResult(
                rock_points=rock_points_array,
                bottom_points=bottom_points,
                basal_indices=basal_indices
            )

        except Exception as e:
            logging.error(f"Error in prepare_bottom_face: {str(e)}")
            raise

    def compute_normals_for_visualization(
        self,
        rock_points: np.ndarray,
        bottom_points: np.ndarray,
        k: Optional[int] = None,
        smooth_iter: int = 0,
    ) -> tuple:
        """
        Computes normals for rock and bottom points, preparing them for visualization.
        
        Args:
            rock_points: Array of rock point coordinates
            bottom_points: Array of bottom face point coordinates
            
        Returns:
            tuple: (rock_pcd, bottom_pcd, combined_points, combined_colors, combined_normals)
        """
        try:
            neighbor_count = max(3, int(k) if k else 200)

            # Use PyMeshLab for enhanced normal computation (with fallback to separate orientation)
            rock_pcd, bottom_pcd = self.compute_normals_pymeshlab(
                rock_points,
                bottom_points,
                k_neighbors=neighbor_count,
                smooth_iter=smooth_iter,
            )
            
            # Prepare visualization data
            combined_points = np.vstack((np.asarray(rock_pcd.points), np.asarray(bottom_pcd.points)))
            combined_colors = np.vstack((
                np.full((len(np.asarray(rock_pcd.points)), 3), [1.0, 0.0, 0.0]),  # Red for rock points
                np.full((len(np.asarray(bottom_pcd.points)), 3), [0.0, 1.0, 0.0])  # Green for bottom face points
            ))
            
            # Combine normals from both point clouds
            combined_normals = np.vstack((np.asarray(rock_pcd.normals), np.asarray(bottom_pcd.normals)))
            
            return rock_pcd, bottom_pcd, combined_points, combined_colors, combined_normals

        except Exception as e:
            logging.error(f"Error in compute_normals_for_visualization: {str(e)}")
            raise

    def compute_normals_for_visualization_separate(
        self,
        rock_points: np.ndarray,
        bottom_points: np.ndarray,
        k: Optional[int] = None,
    ) -> tuple:
        """
        Computes normals using separate orientation method (fallback) for visualization.
        
        Args:
            rock_points: Array of rock point coordinates
            bottom_points: Array of bottom face point coordinates
            
        Returns:
            tuple: (rock_pcd, bottom_pcd, combined_points, combined_colors, combined_normals)
        """
        try:
            neighbor_count = max(3, int(k) if k else 200)

            # Use separate orientation method for normal computation
            rock_pcd, bottom_pcd = self._compute_normals_separate_orientation(
                rock_points,
                bottom_points,
                max_nn=neighbor_count,
            )
           
            # Prepare visualization data
            combined_points = np.vstack((np.asarray(rock_pcd.points), np.asarray(bottom_pcd.points)))
            combined_colors = np.vstack((
                np.full((len(np.asarray(rock_pcd.points)), 3), [1.0, 0.0, 0.0]),  # Red for rock points
                np.full((len(np.asarray(bottom_pcd.points)), 3), [0.0, 1.0, 0.0])  # Green for bottom face points
            ))
            
            # Combine normals from both point clouds
            combined_normals = np.vstack((np.asarray(rock_pcd.normals), np.asarray(bottom_pcd.normals)))
            
            return rock_pcd, bottom_pcd, combined_points, combined_colors, combined_normals

        except Exception as e:
            logging.error(f"Error in compute_normals_for_visualization_separate: {str(e)}")
            raise

    def apply_noise_removal(self, rock_pcd: o3d.geometry.PointCloud, 
                           bottom_pcd: o3d.geometry.PointCloud,
                           adaptive_k: bool = True) -> tuple:
        """
        Applies noise removal to rock points while preserving bottom face points.
        
        Args:
            rock_pcd: Rock point cloud with normals
            bottom_pcd: Bottom face point cloud with normals
            adaptive_k: Whether to use adaptive k_neighbors based on CV (default True)
            
        Returns:
            tuple: (filtered_rock_pcd, bottom_pcd, combined_points, combined_colors, combined_normals)
        """
        try:
            settings = getattr(self, "noise_settings", {})
            base_k_neighbors = max(5, int(settings.get("sor_neighbors", 100)))
            sor_std_ratio = float(settings.get("sor_std_ratio", 2.0))

            # Use adaptive k_neighbors if enabled
            if adaptive_k:
                from utils import estimate_point_density_cv
                points = np.asarray(rock_pcd.points)
                density_info = estimate_point_density_cv(rock_pcd)
                sor_neighbors = density_info['recommended_k']
                logging.info("Adaptive k_neighbors: CV=%.3f, using k=%d", density_info['cv'], sor_neighbors)
            else:
                sor_neighbors = base_k_neighbors
                logging.info("Using fixed k_neighbors=%d", sor_neighbors)

            logging.info("Applying SOR filter to rock points (k=%s, std=%s)", sor_neighbors, sor_std_ratio)
            filtered_rock_pcd, inlier_indices = rock_pcd.remove_statistical_outlier(
                nb_neighbors=sor_neighbors, 
                std_ratio=sor_std_ratio
            )

            # Note: DBSCAN cluster cleanup is handled separately via "Remove Floating Noise" button
            # Optional cluster cleanup to remove floating islands
            if False and settings.get("cluster_cleanup", False):  # Disabled - use Remove Floating Noise button instead
                base_cluster_eps = float(settings.get("cluster_eps", 0.02))
                dbscan_min_points = max(5, int(settings.get("cluster_dbscan_min_points", 20)))
                cluster_min_pct = float(settings.get("cluster_min_pct", 0.01))
                adaptive_dbscan_eps = bool(settings.get("adaptive_dbscan_eps", False))

                # Compute adaptive eps based on point density if enabled
                if adaptive_dbscan_eps and len(points) > 100:
                    # Sample points for density estimation
                    sample_size = min(1000, len(points))
                    sample_indices = np.random.choice(len(points), sample_size, replace=False)
                    sample_points = points[sample_indices]
                    
                    # Build KD-tree and compute mean NN distance
                    tree = cKDTree(sample_points)
                    distances, _ = tree.query(sample_points, k=11)
                    nn_distances = distances[:, 1:].mean(axis=1)
                    mean_nn = np.mean(nn_distances)
                    cv = np.std(nn_distances) / mean_nn if mean_nn > 0 else 0
                    
                    # Use smaller multiplier (1.5x) for cluster cleanup to be more aggressive
                    # This is different from initial SOR which uses 2.0-2.2x
                    cluster_eps = mean_nn * 1.5
                    
                    logging.info(
                        "Adaptive DBSCAN eps: mean_nn=%.1fmm, CV=%.3f, eps=%.3fm (%.1fmm)",
                        mean_nn * 1000, cv, cluster_eps, cluster_eps * 1000
                    )
                else:
                    cluster_eps = base_cluster_eps
                    if adaptive_dbscan_eps:
                        logging.info("Using fixed eps=%.3fm (too few points for adaptive)", cluster_eps)
                    else:
                        logging.info("Using fixed eps=%.3fm (adaptive disabled)", cluster_eps)

                logging.info(
                    "Running cluster cleanup (eps=%.3f, min_points=%d, min_pct=%.2f%%)",
                    cluster_eps,
                    dbscan_min_points,
                    cluster_min_pct * 100,
                )

                # Use the new clean_outliers_dbscan method
                filtered_rock_pcd = self.clean_outliers_dbscan(
                    filtered_rock_pcd,
                    eps=cluster_eps,
                    min_samples=dbscan_min_points,
                    return_inlier_indices=False
                )
            
            # Prepare updated visualization data
            combined_points = np.vstack((np.asarray(filtered_rock_pcd.points), np.asarray(bottom_pcd.points)))
            combined_colors = np.vstack((
                np.full((len(np.asarray(filtered_rock_pcd.points)), 3), [1.0, 0.0, 0.0]),  # Red for rock points
                np.full((len(np.asarray(bottom_pcd.points)), 3), [0.0, 1.0, 0.0])  # Green for bottom face points
            ))
            
            # Update combined normals
            combined_normals = np.vstack((np.asarray(filtered_rock_pcd.normals), np.asarray(bottom_pcd.normals)))
            
            return filtered_rock_pcd, bottom_pcd, combined_points, combined_colors, combined_normals

        except Exception as e:
            logging.error(f"Error in apply_noise_removal: {str(e)}")
            raise

    def complete_mesh_reconstruction(self, rock_pcd: o3d.geometry.PointCloud, 
                                   bottom_pcd: o3d.geometry.PointCloud, 
                                   depth: int = 8) -> o3d.geometry.TriangleMesh:
        """
        Completes the mesh reconstruction using Poisson reconstruction.
        
        Args:
            rock_pcd: Final rock point cloud with normals
            bottom_pcd: Bottom face point cloud with normals
            depth: Depth parameter for Poisson reconstruction
            
        Returns:
            o3d.geometry.TriangleMesh: Reconstructed mesh
        """
        try:
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
            self.last_error_message = ""
            self.reconstructed_mesh = self.poisson_reconstruction(
                new_pcd,
                depth=depth,
                width=0.0,
                scale=1.5,
                linear_fit=False,
            )

            if self.reconstructed_mesh is None:
                if not self.last_error_message:
                    self.last_error_message = "Poisson reconstruction returned no mesh."
                return None

            # Post-process the mesh
            self.reconstructed_mesh.remove_degenerate_triangles()
            self.reconstructed_mesh.remove_duplicated_triangles()
            self.reconstructed_mesh.remove_duplicated_vertices()
            self.reconstructed_mesh.remove_non_manifold_edges()

            # Save temporary mesh for visualization
            with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as temp_file:
                self.temp_mesh_path = temp_file.name
            o3d.io.write_triangle_mesh(self.temp_mesh_path, self.reconstructed_mesh)

            return self.reconstructed_mesh

        except BaseException as e:
            self.last_error_message = str(e)
            logging.error("Error in complete_mesh_reconstruction: %s", e, exc_info=True)
            self.reconstructed_mesh = None
            self.temp_mesh_path = None
            return None

    @staticmethod
    def poisson_worker_entrypoint(conn: Connection, payload: dict) -> None:
        """Run Poisson reconstruction in an isolated process and stream back results."""
        try:
            processor = MeshProcessor()

            rock_points = payload.get('rock_points')
            rock_normals = payload.get('rock_normals')
            bottom_points = payload.get('bottom_points')
            bottom_normals = payload.get('bottom_normals')
            depth = int(payload.get('depth', 8))

            if rock_points is None or rock_normals is None or bottom_points is None or bottom_normals is None:
                conn.send({
                    'success': False,
                    'message': 'Worker payload missing required arrays.',
                })
                return

            rock_points = np.asarray(rock_points)
            rock_normals = np.asarray(rock_normals)
            bottom_points = np.asarray(bottom_points)
            bottom_normals = np.asarray(bottom_normals)

            if rock_points.size == 0 or bottom_points.size == 0:
                conn.send({
                    'success': False,
                    'message': 'Rock or bottom point arrays are empty. Cannot run Poisson.',
                })
                return

            rock_pcd = o3d.geometry.PointCloud()
            rock_pcd.points = o3d.utility.Vector3dVector(rock_points)
            rock_pcd.normals = o3d.utility.Vector3dVector(rock_normals)

            bottom_pcd = o3d.geometry.PointCloud()
            bottom_pcd.points = o3d.utility.Vector3dVector(bottom_points)
            bottom_pcd.normals = o3d.utility.Vector3dVector(bottom_normals)

            mesh = processor.complete_mesh_reconstruction(rock_pcd, bottom_pcd, depth=depth)
            if mesh is None or processor.temp_mesh_path is None:
                conn.send({
                    'success': False,
                    'message': processor.last_error_message or 'Poisson reconstruction failed.'
                })
                return

            conn.send({
                'success': True,
                'mesh_path': processor.temp_mesh_path,
            })
        except BaseException as exc:  # noqa: BLE001
            conn.send({
                'success': False,
                'message': str(exc),
                'traceback': traceback.format_exc(),
            })
        finally:
            try:
                conn.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass

    def poisson_reconstruction(self, pcd: o3d.geometry.PointCloud, 
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

            self.last_error_message = ""
            return mesh

        except BaseException as e:
            # Catch BaseException to intercept SystemExit triggered by Open3D internals
            self.last_error_message = str(e)
            logging.error("Error in Poisson reconstruction: %s", e, exc_info=True)
            return None

    def generate_bottom_face_points(self, pcd: o3d.geometry.PointCloud, 
                                  basal_indices: np.ndarray,
                                  dense_basal_parts: list = None,
                                  dense_basal_parts_is_lateral: list = None,
                                  degree_u: int = 3, 
                                  degree_v: int = 3,
                                  control_points_u: int = 10,
                                  control_points_v: int = 10,
                                  basal_parts_metadata: dict = None) -> np.ndarray:
        """
        Generate bottom face points with support for multiple parts
        
        Args:
            pcd: Open3D PointCloud object
            basal_indices: Array of indices for basal points (legacy format)
            dense_basal_parts: List of dense basal parts (legacy format)
            dense_basal_parts_is_lateral: List of boolean flags indicating which parts are lateral (legacy format)
            degree_u, degree_v: Degrees for NURBS surface
            control_points_u, control_points_v: Number of control points
            basal_parts_metadata: Enhanced basal parts metadata structure (preferred format)
            
        Returns:
            np.ndarray: Generated bottom face points
        """
        try:
            # Extract dense basal parts from enhanced metadata if available
            if basal_parts_metadata is not None and 'parts' in basal_parts_metadata:
                dense_basal_parts = [
                    np.asarray(part.get('dense_points', []), dtype=float)
                    for part in basal_parts_metadata['parts']
                ]
                dense_basal_parts_is_lateral = [part.get('is_lateral', False) for part in basal_parts_metadata['parts']]
                logging.info(f"Using enhanced basal metadata for bottom face generation with {len(dense_basal_parts)} parts")
            logging.info("Starting bottom face generation")
            
            # Check if we're dealing with multiple parts
            if dense_basal_parts:
                bottom_points = self._generate_multi_part_faces(
                    pcd, 
                    dense_basal_parts,
                    dense_basal_parts_is_lateral,
                    degree_u, 
                    degree_v,
                    control_points_u,
                    control_points_v
                )
            else:
                bottom_points = self._generate_single_face(
                    pcd,
                    basal_indices,
                    degree_u,
                    degree_v,
                    control_points_u,
                    control_points_v
                )
            
            # Apply downsampling to match rock point density and improve uniformity
            if bottom_points is not None:
                bottom_points = self._downsample_bottom_face(pcd, bottom_points, basal_indices)
            
            return bottom_points
                
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
                basal_points = np.asarray(points_or_indices, dtype=float)
                logging.debug(f"Using dense points: input shape={np.asarray(points_or_indices).shape}, output shape={basal_points.shape}")
            else:
                basal_points = points[np.asarray(points_or_indices, dtype=int)]  # Convert indices to points
                logging.debug(f"Using indices: {len(points_or_indices)} indices, output shape={basal_points.shape}")
            
            # Calculate transformation matrix for 2D projection
            center = np.mean(basal_points, axis=0)
            centered_points = basal_points - center
            logging.debug(f"basal_points shape: {basal_points.shape}, center shape: {center.shape}, centered_points shape: {centered_points.shape}")
            
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
                surf.delta = 0.04  # Finer sampling
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

    def compute_normals_pymeshlab(self, rock_points: np.ndarray, bottom_points: np.ndarray,
                                  k_neighbors: int = 200, smooth_iter: int = 0) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
        """
        Compute normals using PyMeshLab for enhanced normal estimation
        
        Args:
            rock_points: Array of rock points
            bottom_points: Array of bottom face points
            k_neighbors: Number of neighbors for normal computation
            smooth_iter: Number of smoothing iterations
            
        Returns:
            Tuple of (rock_pcd, bottom_pcd) with computed normals
        """
        try:
            if not PYMESHLAB_AVAILABLE:
                logging.warning("PyMeshLab not available. Falling back to separate orientation method.")
                return self._compute_normals_separate_orientation(rock_points, bottom_points, max_nn=k_neighbors)
            
            logging.info("Computing normals using PyMeshLab...")
            
            # Combine all points
            combined_points = np.vstack([rock_points, bottom_points])
            
            # Create MeshSet
            ms = pymeshlab.MeshSet()
            
            # Create mesh from points
            mesh = pymeshlab.Mesh(vertex_matrix=combined_points)
            ms.add_mesh(mesh)
            
            logging.info(f"Computing normals with k={k_neighbors}, smooth_iter={smooth_iter}")
            
            # Compute normals using PyMeshLab
            ms.compute_normal_for_point_clouds(k=k_neighbors, smoothiter=smooth_iter)
            
            # Get the mesh with computed normals
            processed_mesh = ms.current_mesh()
            
            # Extract vertices and normals
            vertices = processed_mesh.vertex_matrix()
            normals = processed_mesh.vertex_normal_matrix()
            
            # Check normal orientation and flip if majority point towards center
            logging.info("Checking normal orientation...")
            center = np.mean(vertices, axis=0)
            directions = vertices - center
            directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]
            
            # Compute dot products to check orientation
            dots = np.sum(directions * normals, axis=1)
            inward_pointing = np.sum(dots < 0)
            outward_pointing = np.sum(dots >= 0)
            
            logging.info(f"Normals pointing inward: {inward_pointing}, outward: {outward_pointing}")
            
            # If majority of normals point inward (towards center), flip them all
            if inward_pointing > outward_pointing:
                logging.info("Majority of normals point inward, flipping all normals to point outward")
                normals = -normals
            else:
                logging.info("Majority of normals already point outward, keeping orientation")
            
            # Split back into rock and bottom point clouds
            rock_pcd = o3d.geometry.PointCloud()
            rock_pcd.points = o3d.utility.Vector3dVector(vertices[:len(rock_points)])
            rock_pcd.normals = o3d.utility.Vector3dVector(normals[:len(rock_points)])
            
            bottom_pcd = o3d.geometry.PointCloud()
            bottom_pcd.points = o3d.utility.Vector3dVector(vertices[len(rock_points):])
            bottom_pcd.normals = o3d.utility.Vector3dVector(normals[len(rock_points):])
            
            logging.info(f"PyMeshLab normal computation completed for {len(combined_points)} points")
            return rock_pcd, bottom_pcd
            
        except Exception as e:
            logging.error(f"Error in PyMeshLab normal computation: {str(e)}")
            logging.info("Falling back to separate orientation method...")
            return self._compute_normals_separate_orientation(rock_points, bottom_points, max_nn=k_neighbors)

    def _compute_normals_separate_orientation(
        self,
        rock_points: np.ndarray,
        bottom_points: np.ndarray,
        max_nn: int = 200,
    ) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
        """
        Fallback method: Compute normals using separate orientation (current method)
        
        Args:
            rock_points: Array of rock points
            bottom_points: Array of bottom face points
            
        Returns:
            Tuple of (rock_pcd, bottom_pcd) with computed normals
        """
        try:
            logging.info("Using separate orientation method for normal computation...")
            
            # Create separate point clouds
            rock_pcd = o3d.geometry.PointCloud()
            rock_pcd.points = o3d.utility.Vector3dVector(rock_points)
            
            bottom_pcd = o3d.geometry.PointCloud()
            bottom_pcd.points = o3d.utility.Vector3dVector(bottom_points)
            
            # Estimate normals with better parameters for rock points
            rock_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=max_nn)
            )
            rock_consistency = max(3, min(max_nn, len(rock_points)))
            rock_pcd.orient_normals_consistent_tangent_plane(rock_consistency)
            
            # Estimate normals for bottom points
            bottom_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=max_nn)
            )
            
            # Reorient normals for rock_pcd
            logging.info("Reorienting normals for rock_pcd...")
            rock_center = rock_pcd.get_center()
            rock_points_array = np.asarray(rock_pcd.points)
            rock_normals = np.asarray(rock_pcd.normals)
            rock_directions = rock_points_array - rock_center
            rock_directions = rock_directions / np.linalg.norm(rock_directions, axis=1)[:, np.newaxis]
            rock_dots = np.sum(rock_directions * rock_normals, axis=1)
            rock_normals[rock_dots < 0] = -rock_normals[rock_dots < 0]
            rock_pcd.normals = o3d.utility.Vector3dVector(rock_normals)

            # Reorient normals for bottom_pcd
            logging.info("Reorienting normals for bottom_pcd...")
            bottom_center = bottom_pcd.get_center()
            bottom_points_array = np.asarray(bottom_pcd.points)
            bottom_normals = np.asarray(bottom_pcd.normals)
            bottom_directions = bottom_points_array - bottom_center
            bottom_directions = bottom_directions / np.linalg.norm(bottom_directions, axis=1)[:, np.newaxis]
            bottom_dots = np.sum(bottom_directions * bottom_normals, axis=1)
            bottom_normals[bottom_dots < 0] = -bottom_normals[bottom_dots < 0]
            bottom_pcd.normals = o3d.utility.Vector3dVector(bottom_normals)
            bottom_consistency = max(3, min(max_nn, len(bottom_points)))
            bottom_pcd.orient_normals_consistent_tangent_plane(bottom_consistency)
            
            return rock_pcd, bottom_pcd
            
        except Exception as e:
            logging.error(f"Error in separate orientation normal computation: {str(e)}")
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

    def _downsample_bottom_face(self, pcd: o3d.geometry.PointCloud, 
                              bottom_points: np.ndarray, 
                              basal_indices: np.ndarray) -> np.ndarray:
        """
        Downsample the bottom face points to match rock point density and improve uniformity.
        
        Args:
            pcd: Original point cloud (for rock density reference)
            bottom_points: Dense NURBS-generated bottom face points
            basal_indices: Indices of basal points (to preserve boundary)
            
        Returns:
            np.ndarray: Downsampled bottom face points
        """
        try:
            # Get rock points for density analysis
            points = np.asarray(pcd.points)
            rock_mask = np.ones(len(points), dtype=bool)
            if len(basal_indices) > 0:
                rock_mask[basal_indices] = False
            rock_points = points[rock_mask]
            
            # Calculate target density based on rock points
            if len(rock_points) > 1:
                # Use nearest neighbor distances to estimate density
                rock_pcd = o3d.geometry.PointCloud()
                rock_pcd.points = o3d.utility.Vector3dVector(rock_points)
                
                # Calculate average nearest neighbor distance in rock points
                distances = rock_pcd.compute_nearest_neighbor_distance()
                avg_distance = np.mean(distances)
                target_voxel_size = avg_distance * 0.8  # Slightly denser to ensure good coverage
                
                logging.info(f"Rock point avg distance: {avg_distance:.4f}, target voxel size: {target_voxel_size:.4f}")
            else:
                # Fallback if no rock points
                target_voxel_size = 0.05
                logging.warning("No rock points found, using default voxel size")
            
            # Create point cloud from bottom points
            bottom_pcd = o3d.geometry.PointCloud()
            bottom_pcd.points = o3d.utility.Vector3dVector(bottom_points)
            
            logging.info(f"Bottom face points before downsampling: {len(bottom_points)}")
            
            # Apply voxel downsampling
            downsampled_pcd = bottom_pcd.voxel_down_sample(voxel_size=target_voxel_size)
            downsampled_points = np.asarray(downsampled_pcd.points)
            
            logging.info(f"Bottom face points after voxel downsampling: {len(downsampled_points)}")
            
            # If we have basal points, ensure they're preserved in the downsampled set
            if len(basal_indices) > 0:
                basal_points = points[basal_indices]
                
                # Find which basal points might have been removed during downsampling
                tree = cKDTree(downsampled_points)
                distances, _ = tree.query(basal_points)
                
                # Add back basal points that are too far from downsampled points
                missing_basal = basal_points[distances > target_voxel_size]
                if len(missing_basal) > 0:
                    logging.info(f"Adding back {len(missing_basal)} basal points that were lost during downsampling")
                    downsampled_points = np.vstack([downsampled_points, missing_basal])
            
            # Optional: Apply uniform downsampling if still too dense
            if len(downsampled_points) > len(rock_points) * 1.5:  # If still 50% denser than rock
                # Use random downsampling to match rock density
                target_count = int(len(rock_points) * 1.2)  # 20% denser than rock
                if target_count < len(downsampled_points):
                    indices = np.random.choice(len(downsampled_points), target_count, replace=False)
                    downsampled_points = downsampled_points[indices]
                    logging.info(f"Applied random downsampling to {len(downsampled_points)} points")
            
            logging.info(f"Final bottom face points after downsampling: {len(downsampled_points)}")
            return downsampled_points
            
        except Exception as e:
            logging.error(f"Error in bottom face downsampling: {str(e)}")
            logging.warning("Returning original bottom points without downsampling")
            return bottom_points
