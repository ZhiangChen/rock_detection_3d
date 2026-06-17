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

    def get_epsg_code(self, las_file_path: Union[str, Path]) -> Optional[int]:
        """Get EPSG code from LAS/LAZ file."""
        try:
            pc = laspy.read(las_file_path)
            crs = pc.header.parse_crs()
            if crs is None:
                # logging.warning(f"No CRS information found in {las_file_path}")
                return None
            epsg = crs.to_epsg()
            logging.info(f"Found EPSG code: {epsg}")
            return epsg
        except Exception as e:
            logging.error(f"Error getting EPSG code: {e}")
            return None

    def load_las_as_open3d_point_cloud(self, las_file_path: Union[str, Path], 
                                     evaluate: bool = False) -> Tuple[o3d.geometry.PointCloud, Optional[np.ndarray], Optional[int]]:
        """
        Load a LAS/LAZ file and convert it to an Open3D point cloud.
        
        Args:
            las_file_path: Path to the LAS/LAZ file
            evaluate: Boolean indicating if ground truth labels should be loaded
            
        Returns:
            tuple: (Open3D PointCloud object, ground truth labels if available, EPSG code if available)
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
                # Auto-detect color range: 16-bit (0-65535) or 8-bit (0-255)
                red_max = pc.red.max()
                green_max = pc.green.max()
                blue_max = pc.blue.max()
                color_max = max(red_max, green_max, blue_max)
                
                if color_max <= 255:
                    # 8-bit colors (0-255 range)
                    r = np.uint8(pc.red)
                    g = np.uint8(pc.green)
                    b = np.uint8(pc.blue)
                    logging.info(f"Detected 8-bit color range (max={color_max})")
                else:
                    # 16-bit colors (0-65535 range) - normalize to 8-bit
                    r = np.uint8(pc.red / 65535.0 * 255)
                    g = np.uint8(pc.green / 65535.0 * 255)
                    b = np.uint8(pc.blue / 65535.0 * 255)
                    logging.info(f"Detected 16-bit color range (max={color_max}), normalizing to 8-bit")
                
                rgb = np.vstack((r, g, b)).transpose() / 255.0
            else:
                rgb = np.zeros((len(x), 3))

            # Create Open3D PointCloud object and set points and colors
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(rgb)

            epsg_code = self.get_epsg_code(las_file_path)
            return pcd, ground_truth_labels, epsg_code
            
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
                        basal_data: Optional[Union[np.ndarray, list, dict]] = None,
                        plain: bool = False) -> str:
        """
        Save point cloud to a LAS file with optional color-coded labels and enhanced basal point encoding.
        
        Args:
            pcd: Open3D PointCloud object to save
            file_path: Path where to save the point cloud
            labels: Optional array of labels for coloring (0 for pedestal, 1 for rock)
            basal_data: Can be:
                - Legacy: array of basal point indices or boolean mask
                - Enhanced: dictionary with basal parts metadata structure
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

            # Initialize intensity and classification arrays
            intensity = np.zeros_like(x, dtype=np.uint16)
            classification = np.zeros_like(x, dtype=np.uint8)  # Add classification field

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
                    
                    # Set intensity and classification based on labels
                    intensity[rock_mask] = 1
                    intensity[pedestal_mask] = 0
                    classification[rock_mask] = 1  # Rock classification
                    classification[pedestal_mask] = 2  # Pedestal classification

                # Enhanced basal points handling with part information
                if basal_data is not None:
                    self._encode_basal_points(basal_data, points, red, green, blue, intensity, classification)

            # Set colors, intensity, and classification in LAS file
            las.red = red
            las.green = green
            las.blue = blue
            las.intensity = intensity
            las.classification = classification

            # Save the file
            las.write(file_path)
            logging.info(f"Point cloud saved to {file_path}")
            return str(file_path)

        except Exception as e:
            logging.error(f"Error saving point cloud: {e}")
            raise

    def _encode_basal_points(self, basal_data, points, red, green, blue, intensity, classification):
        """
        Enhanced basal point encoding with part information.
        
        Encoding scheme:
        - Intensity: part_id + (100 * is_lateral)
        - Classification: 9 for regular basal, 10 for lateral basal (LAS-compliant codes)
        - Colors: Distinct per part, lighter for lateral
        """
        if isinstance(basal_data, dict) and 'parts' in basal_data:
            # Enhanced metadata structure
            self._encode_structured_basal_data(basal_data, points, red, green, blue, intensity, classification)
        else:
            # Legacy fallback: treat as simple basal points
            self._encode_legacy_basal_points(basal_data, points, red, green, blue, intensity, classification)

    def _encode_structured_basal_data(self, basal_metadata, points, red, green, blue, intensity, classification):
        """Encode structured basal parts metadata."""
        total_basal_points = 0
        
        for part in basal_metadata['parts']:
            part_id = part['id']
            is_lateral = part['is_lateral']
            point_indices = part['point_indices']
            part_color = part.get('color', [0, 1, 0])  # Default to green
            
            if len(point_indices) > 0:
                # Intensity encoding: part_id + (100 * is_lateral)
                intensity_value = part_id + (100 if is_lateral else 0)
                intensity[point_indices] = intensity_value
                
                # Classification encoding: Use valid LAS classification codes
                # Class 9 = Water (repurposed for regular basal points)
                # Class 10 = Rail (repurposed for lateral basal points)
                classification_value = 10 if is_lateral else 9
                classification[point_indices] = classification_value
                
                # Color encoding: distinct per part, lighter for lateral
                color_multiplier = 65535
                if is_lateral:
                    # Make lateral points lighter
                    part_color = [min(1.0, c + 0.3) for c in part_color]
                
                red[point_indices] = int(part_color[0] * color_multiplier)
                green[point_indices] = int(part_color[1] * color_multiplier)
                blue[point_indices] = int(part_color[2] * color_multiplier)
                
                total_basal_points += len(point_indices)
                
                part_type = "lateral" if is_lateral else "regular"
                logging.info(f"Encoded part {part_id} ({part_type}): {len(point_indices)} points with intensity {intensity_value}, class {classification_value}")
        
        logging.info(f"Total basal points encoded: {total_basal_points} across {len(basal_metadata['parts'])} parts")

    def _encode_legacy_basal_points(self, basal_points, points, red, green, blue, intensity, classification):
        """Legacy encoding for simple basal points (backward compatibility)."""
        if isinstance(basal_points, (list, np.ndarray)) and len(basal_points) > 0:
            if not isinstance(basal_points, np.ndarray) or basal_points.dtype != bool:
                basal_mask = np.zeros(len(points), dtype=bool)
                basal_mask[basal_points] = True
            else:
                basal_mask = basal_points
            
            # Legacy encoding: green color, intensity 2, classification 9 (regular basal)
            red[basal_mask] = 0
            green[basal_mask] = 65535
            blue[basal_mask] = 0
            intensity[basal_mask] = 2
            classification[basal_mask] = 9  # Regular basal classification (repurposed from Water)
            
            basal_count = np.sum(basal_mask)
            logging.info(f"Legacy encoding: {basal_count} basal points with intensity 2, class 9")

    @staticmethod
    def get_part_color(part_id: int) -> list:
        """Get a distinct color for a given part ID."""
        colors = [
            [1, 0, 0],    # Red
            [0, 1, 0],    # Green  
            [0, 0, 1],    # Blue
            [1, 1, 0],    # Yellow
            [1, 0, 1],    # Magenta
            [0, 1, 1],    # Cyan
            [1, 0.5, 0],  # Orange
            [0.5, 0, 1],  # Purple
        ]
        return colors[(part_id - 1) % len(colors)]

    @staticmethod
    def decode_basal_part_info(intensity_value: int) -> tuple:
        """
        Decode part information from intensity value.
        
        Args:
            intensity_value: Encoded intensity value
            
        Returns:
            tuple: (part_id, is_lateral)
        """
        is_lateral = intensity_value >= 100
        part_id = intensity_value % 100 if is_lateral else intensity_value
        return part_id, is_lateral

    def load_basal_parts_from_las(self, las_file_path: Union[str, Path]) -> dict:
        """
        Load and decode basal parts information from a saved LAS file.
        
        Args:
            las_file_path: Path to the LAS file with encoded basal parts
            
        Returns:
            dict: Decoded basal parts metadata structure
        """
        try:
            pc = laspy.read(las_file_path)
            
            # Find basal points (classification 9 or 10, and legacy support for intensity 2)
            basal_mask = (pc.classification == 9) | (pc.classification == 10) | (pc.intensity == 2)
            basal_indices = np.where(basal_mask)[0]
            
            if len(basal_indices) == 0:
                return {'parts': [], 'num_parts': 0, 'has_lateral_parts': False}
            
            # Extract basal point data
            basal_intensities = pc.intensity[basal_indices]
            basal_classifications = pc.classification[basal_indices]
            
            # Group by part ID
            parts_dict = {}
            for i, (intensity, classification) in enumerate(zip(basal_intensities, basal_classifications)):
                # Handle different encoding schemes
                if intensity == 2 and classification != 9 and classification != 10:
                    # Legacy encoding: single part, not lateral
                    part_id = 1
                    is_lateral = False
                elif classification == 9 or classification == 10:
                    # New encoding scheme
                    part_id, is_lateral = self.decode_basal_part_info(intensity)
                    # Verify classification matches intensity encoding
                    expected_class = 10 if is_lateral else 9
                    if classification != expected_class:
                        logging.warning(f"Classification mismatch: intensity={intensity}, class={classification}, expected={expected_class}")
                else:
                    # Fallback decoding
                    part_id, is_lateral = self.decode_basal_part_info(intensity)
                
                if part_id not in parts_dict:
                    parts_dict[part_id] = {
                        'id': part_id,
                        'is_lateral': is_lateral,
                        'point_indices': [],
                        'coordinates': []
                    }
                
                parts_dict[part_id]['point_indices'].append(basal_indices[i])
                idx = int(basal_indices[i])  # Convert numpy int64 to Python int
                parts_dict[part_id]['coordinates'].append([pc.x[idx], pc.y[idx], pc.z[idx]])
            
            # Convert to list and add metadata
            parts_list = []
            for part_id in sorted(parts_dict.keys()):
                part = parts_dict[part_id]
                part['point_indices'] = np.array(part['point_indices'])
                part['coordinates'] = np.array(part['coordinates'])
                part['num_points'] = len(part['point_indices'])
                parts_list.append(part)
            
            metadata = {
                'parts': parts_list,
                'num_parts': len(parts_list),
                'has_lateral_parts': any(part['is_lateral'] for part in parts_list),
                'total_basal_points': len(basal_indices)
            }
            
            logging.info(f"Loaded {len(parts_list)} basal parts with {len(basal_indices)} total points")
            return metadata
            
        except Exception as e:
            logging.error(f"Error loading basal parts from LAS file: {e}")
            raise

    def extract_parts_for_mesh_reconstruction(self, las_file_path: Union[str, Path]) -> dict:
        """
        Extract basal parts information in a format optimized for external mesh reconstruction.
        
        Args:
            las_file_path: Path to the LAS file
            
        Returns:
            dict: Mesh reconstruction ready data structure
        """
        basal_metadata = self.load_basal_parts_from_las(las_file_path)
        
        reconstruction_data = {
            'basal_parts': [],
            'lateral_parts': [],
            'metadata': {
                'num_basal_parts': 0,
                'num_lateral_parts': 0,
                'total_parts': basal_metadata['num_parts'],
                'coordinate_system': None  # Could be enhanced with CRS info
            }
        }
        
        for part in basal_metadata['parts']:
            part_data = {
                'part_id': part['id'],
                'coordinates': part['coordinates'],
                'num_points': part['num_points'],
                'point_indices': part['point_indices']
            }
            
            if part['is_lateral']:
                reconstruction_data['lateral_parts'].append(part_data)
            else:
                reconstruction_data['basal_parts'].append(part_data)
        
        reconstruction_data['metadata']['num_basal_parts'] = len(reconstruction_data['basal_parts'])
        reconstruction_data['metadata']['num_lateral_parts'] = len(reconstruction_data['lateral_parts'])
        
        return reconstruction_data

    def export_parts_to_separate_files(self, las_file_path: Union[str, Path], output_dir: Union[str, Path]) -> list:
        """
        Export each basal part to a separate file for individual processing.
        
        Args:
            las_file_path: Path to the source LAS file
            output_dir: Directory to save separate part files
            
        Returns:
            list: List of created file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        basal_metadata = self.load_basal_parts_from_las(las_file_path)
        created_files = []
        
        # Load full point cloud
        pc = laspy.read(las_file_path)
        
        for part in basal_metadata['parts']:
            part_type = "lateral" if part['is_lateral'] else "basal"
            filename = f"part_{part['id']:02d}_{part_type}.las"
            output_path = output_dir / filename
            
            # Create new LAS file with just this part's points
            header = laspy.LasHeader(point_format=3, version="1.2")
            part_las = laspy.LasData(header)
            
            indices = part['point_indices']
            part_las.x = pc.x[indices]
            part_las.y = pc.y[indices]
            part_las.z = pc.z[indices]
            part_las.intensity = pc.intensity[indices]
            part_las.classification = pc.classification[indices]
            
            if hasattr(pc, 'red'):
                part_las.red = pc.red[indices]
                part_las.green = pc.green[indices]
                part_las.blue = pc.blue[indices]
            
            part_las.write(output_path)
            created_files.append(str(output_path))
            logging.info(f"Exported part {part['id']} ({part_type}) to {output_path}")
        
        return created_files

    def generate_reconstruction_report(self, las_file_path: Union[str, Path]) -> str:
        """
        Generate a human-readable report of basal parts for reconstruction planning.
        
        Args:
            las_file_path: Path to the LAS file
            
        Returns:
            str: Formatted report
        """
        basal_metadata = self.load_basal_parts_from_las(las_file_path)
        
        report = ["BASAL PARTS RECONSTRUCTION REPORT", "=" * 40, ""]
        report.append(f"Total Parts: {basal_metadata['num_parts']}")
        report.append(f"Total Basal Points: {basal_metadata['total_basal_points']}")
        report.append(f"Has Lateral Parts: {'Yes' if basal_metadata['has_lateral_parts'] else 'No'}")
        report.append("")
        
        for part in basal_metadata['parts']:
            part_type = "LATERAL" if part['is_lateral'] else "BASAL CONTACT"
            report.append(f"Part {part['id']} ({part_type}):")
            report.append(f"  - Points: {part['num_points']}")
            report.append(f"  - Intensity Range: {part['point_indices'].min()}-{part['point_indices'].max()}")
            
            # Calculate bounding box
            coords = part['coordinates']
            bbox = {
                'min_x': coords[:, 0].min(), 'max_x': coords[:, 0].max(),
                'min_y': coords[:, 1].min(), 'max_y': coords[:, 1].max(),
                'min_z': coords[:, 2].min(), 'max_z': coords[:, 2].max(),
            }
            report.append(f"  - Bounding Box: X[{bbox['min_x']:.2f}, {bbox['max_x']:.2f}] Y[{bbox['min_y']:.2f}, {bbox['max_y']:.2f}] Z[{bbox['min_z']:.2f}, {bbox['max_z']:.2f}]")
            report.append("")
        
        return "\n".join(report)

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