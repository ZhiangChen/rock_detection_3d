import open3d as o3d
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Union
from PyQt5.QtWidgets import QDialog, QSpinBox, QVBoxLayout, QPushButton, QLabel
from basal_points_algo import BasalPointAlgorithm

class BasalLineProcessor:
    """
    Handles all basal line related operations including point selection,
    estimation, and multi-part processing.
    """
    
    def __init__(self):
        self.poc_points = []  # Points of contact
        self.basal_points = None  # Final basal points
        self.basal_parts = []  # Store parts for multi-part processing
        self.current_part = 0  # Track current part in multi-part processing
        self.num_parts = 0  # Total number of parts
        self.dense_basal_parts = []  # Store dense points for each part

    def start_multi_part_input(self, num_parts: int, dialog: QDialog) -> None:
        """
        Initialize multi-part input process.
        
        Args:
            num_parts: Number of parts to process
            dialog: QDialog to close after initialization
        """
        logging.info(f"Starting multi-part input with {num_parts} parts")
        try:
            self.num_parts = num_parts
            self.current_part = 0
            self.basal_parts = []
            self.dense_basal_parts = []
            dialog.accept()
            
        except Exception as e:
            logging.error(f"Error initializing multi-part input: {str(e)}")
            raise

    def process_current_part(self, selected_points: List[int], pcd: o3d.geometry.PointCloud) -> Tuple[bool, str]:
        """
        Process the points selected for current part.
        
        Args:
            selected_points: List of selected point indices
            pcd: Open3D PointCloud object
            
        Returns:
            Tuple[bool, str]: (Success status, Message to display)
        """
        try:
            if selected_points is not None and len(selected_points) > 0:
                # Store selected points for this part
                self.basal_parts.append(selected_points)
                
                # Generate dense points for this part
                points = np.asarray(pcd.points)
                algorithm = BasalPointAlgorithm(pcd)
                dense_points = algorithm.generate_dense_basal_points(points[selected_points])
                self.dense_basal_parts.append(dense_points)
                
                self.current_part += 1
                
                # Check if we've processed all parts
                if self.current_part >= self.num_parts:
                    self.basal_points = np.concatenate(self.basal_parts)
                    return True, "All parts processed successfully"
                else:
                    return False, f"Part {self.current_part} of {self.num_parts} completed. Please select points for the next part."
            else:
                return False, "No points were selected. Please try again."
                
        except Exception as e:
            logging.error(f"Error processing part {self.current_part + 1}: {str(e)}")
            return False, f"Error processing part {self.current_part + 1}: {str(e)}"

    def estimate_basal_points(self, pcd: o3d.geometry.PointCloud) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        Estimate basal points based on the selected points of contact.
        
        Args:
            pcd: Open3D PointCloud object
            
        Returns:
            Tuple containing:
                - np.ndarray: Estimated basal points
                - List[np.ndarray]: Dense basal points for each part
                - List[np.ndarray]: Part colors for visualization
        """
        try:
            # Create colors for different parts
            part_colors = [
                [0, 1, 0],     # Green
                [1, 1, 0],     # Yellow
                [0, 1, 1],     # Cyan
                [1, 0, 1],     # Magenta
                [0.5, 0.5, 0], # Olive
            ]
            
            # Handle multi-part basal points
            if hasattr(self, 'basal_parts') and len(self.basal_parts) > 1:
                all_basal_points = []
                dense_points_list = []
                colors_list = []
                
                algorithm = BasalPointAlgorithm(pcd)
                
                for i, part_points in enumerate(self.basal_parts):
                    points = np.asarray(pcd.points)[part_points]
                    dense_points = algorithm.generate_dense_basal_points(points)
                    
                    all_basal_points.extend(part_points)
                    dense_points_list.append(dense_points)
                    colors_list.append(part_colors[i % len(part_colors)])
                
                self.basal_points = np.array(all_basal_points)
                self.dense_basal_parts = dense_points_list
                
                return self.basal_points, dense_points_list, colors_list
            
            # Handle single part basal points
            else:
                if not self.poc_points or len(self.poc_points) == 0:
                    raise ValueError("No points of contact selected")
                
                algorithm = BasalPointAlgorithm(pcd)
                points = np.asarray(pcd.points)[self.poc_points]
                dense_points = algorithm.generate_dense_basal_points(points)
                
                self.basal_points = np.array(self.poc_points)
                self.dense_basal_parts = [dense_points]
                
                return self.basal_points, [dense_points], [[0, 1, 0]]  # Single part, green color
                
        except Exception as e:
            logging.error(f"Error in basal points estimation: {str(e)}")
            raise

    def get_basal_points(self) -> Optional[np.ndarray]:
        """
        Get the current basal points.
        
        Returns:
            Optional[np.ndarray]: Current basal points or None if not set
        """
        return self.basal_points if self.basal_points is not None else None

    def get_dense_basal_parts(self) -> List[np.ndarray]:
        """
        Get the dense basal points for all parts.
        
        Returns:
            List[np.ndarray]: List of dense basal points for each part
        """
        return self.dense_basal_parts

    def reset(self) -> None:
        """Reset all basal line processing data."""
        self.poc_points = []
        self.basal_points = None
        self.basal_parts = []
        self.current_part = 0
        self.num_parts = 0
        self.dense_basal_parts = []