import open3d as o3d
import numpy as np
import laspy
import logging
from pathlib import Path
import csv
from typing import Tuple, Optional, Union

class PointCloudFileHandler:
    """
    Handles all file I/O operations for point clouds, including loading and saving
    point cloud data in various formats.
    """
    
    def __init__(self):
        self.x_mean = 0
        self.y_mean = 0
        self.z_mean = 0

    def load_las_as_open3d_point_cloud(self, las_file_path: Union[str, Path], 
                                     evaluate: bool = False) -> Tuple[o3d.geometry.PointCloud, Optional[np.ndarray]]:
        """
        Load a LAS/LAZ file and convert it to an Open3D point cloud.
        
        Args:
            las_file_path: Path to the LAS/LAZ file
            evaluate: Boolean indicating if ground truth labels should be loaded
            
        Returns:
            tuple: (Open3D PointCloud object, ground truth labels if available)
        """
        try:
            # Read LAS/LAZ file using laspy
            pc = laspy.read(las_file_path)
            x, y, z = pc.x, pc.y, pc.z
            ground_truth_labels = None

            # Check if ground truth labels are available for evaluation
            if evaluate and "Original cloud index" in pc.point_format.dimension_names:
                ground_truth_labels = np.int_(pc["Original cloud index"])

            # Store the mean values for recentering later
            self.x_mean = np.mean(x)
            self.y_mean = np.mean(y)
            self.z_mean = np.mean(z)

            # Recenter the point cloud
            xyz = np.vstack((x - self.x_mean, y - self.y_mean, z - self.z_mean)).transpose()

            # Check if RGB color information is available in the LAS file
            if all(dim in pc.point_format.dimension_names for dim in ["red", "green", "blue"]):
                r = np.uint8(pc.red / 65535.0 * 255)
                g = np.uint8(pc.green / 65535.0 * 255)
                b = np.uint8(pc.blue / 65535.0 * 255)
                rgb = np.vstack((r, g, b)).transpose() / 255.0
            else:
                rgb = np.zeros((len(x), 3))

            # Create Open3D PointCloud object and set points and colors
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(rgb)

            return pcd, ground_truth_labels
            
        except laspy.errors.LaspyException as e:
            if "No LazBackend selected" in str(e):
                logging.error("Error: Unable to read LAZ file. Please ensure 'lazrs' is installed.")
                logging.error("Run: pip install lazrs")
                raise
            else:
                logging.error(f"Error reading point cloud file: {e}")
                raise
        except Exception as e:
            logging.error(f"Unexpected error reading point cloud file: {e}")
            raise

    def save_point_cloud(self, pcd: o3d.geometry.PointCloud, 
                        file_path: Union[str, Path], 
                        labels: Optional[np.ndarray] = None,
                        basal_points: Optional[Union[np.ndarray, list]] = None,
                        plain: bool = False) -> str:
        """
        Save point cloud to a LAS file with optional color-coded labels.
        
        Args:
            pcd: Open3D PointCloud object to save
            file_path: Path where to save the point cloud
            labels: Optional array of labels for coloring (0 for pedestal, 1 for rock)
            basal_points: Optional array of basal point indices or boolean mask
            plain: If True, saves all points in red without classification colors
            
        Returns:
            str: Path to the saved file
        """
        try:
            file_path = Path(file_path)
            if not str(file_path).lower().endswith('.las'):
                file_path = file_path.with_suffix('.las')

            # Create a new LAS file
            header = laspy.LasHeader(point_format=3, version="1.2")  # Changed to format 3 to include intensity
            las = laspy.LasData(header)

            # Get points and restore original coordinates
            points = np.asarray(pcd.points)
            x = points[:, 0] + self.x_mean
            y = points[:, 1] + self.y_mean
            z = points[:, 2] + self.z_mean

            # Set coordinates
            las.x = x
            las.y = y
            las.z = z

            # Initialize intensity array
            intensity = np.zeros_like(x, dtype=np.uint16)

            # Set colors and intensity based on classification
            if plain:
                red = np.full_like(x, 65535, dtype=np.uint16)
                green = np.zeros_like(x, dtype=np.uint16)
                blue = np.zeros_like(x, dtype=np.uint16)
            else:
                red = np.zeros_like(x, dtype=np.uint16)
                green = np.zeros_like(x, dtype=np.uint16)
                blue = np.zeros_like(x, dtype=np.uint16)

                if labels is not None:
                    # Color coding: Blue for pedestal (0), Red for rock (1)
                    rock_mask = labels == 1
                    pedestal_mask = labels == 0
                    
                    red[rock_mask] = 65535  # Red for rock
                    blue[pedestal_mask] = 65535  # Blue for pedestal
                    
                    # Set intensity based on labels
                    intensity[rock_mask] = 1
                    intensity[pedestal_mask] = 0

                # Handle basal points (green color)
                if basal_points is not None:
                    # Convert basal_points to boolean mask if it's an index array
                    if isinstance(basal_points, (list, np.ndarray)):
                        if len(basal_points) > 0:
                            if not isinstance(basal_points, np.ndarray) or basal_points.dtype != bool:
                                basal_mask = np.zeros(len(points), dtype=bool)
                                basal_mask[basal_points] = True
                            else:
                                basal_mask = basal_points
                            
                            # Set green color for basal points
                            red[basal_mask] = 0
                            green[basal_mask] = 65535
                            blue[basal_mask] = 0
                            
                            # Set intensity value 2 for basal points
                            intensity[basal_mask] = 2
                            
                            basal_count = np.sum(basal_mask)
                            logging.info(f"Labeled {basal_count} basal points in the point cloud.")

            # Set colors and intensity in LAS file
            las.red = red
            las.green = green
            las.blue = blue
            las.intensity = intensity

            # Save the file
            las.write(file_path)
            logging.info(f"Point cloud saved to {file_path}")
            return str(file_path)

        except Exception as e:
            logging.error(f"Error saving point cloud: {e}")
            raise

    def save_results_csv(self, results: dict, csv_path: Union[str, Path]) -> None:
        """
        Save analysis results to a CSV file.
        
        Args:
            results: Dictionary containing analysis results
            csv_path: Path to the CSV file
        """
        try:
            csv_path = Path(csv_path)
            
            # Convert numpy arrays to lists for CSV storage
            results_to_save = {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in results.items()
            }

            # Write to CSV
            if not csv_path.exists():
                with open(csv_path, mode='w', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=results_to_save.keys())
                    writer.writeheader()
                    writer.writerow(results_to_save)
            else:
                with open(csv_path, mode='a', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=results_to_save.keys())
                    writer.writerow(results_to_save)

            logging.info(f"Results saved to {csv_path}")

        except Exception as e:
            logging.error(f"Error saving results to CSV: {e}")
            raise