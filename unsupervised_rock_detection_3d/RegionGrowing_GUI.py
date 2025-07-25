import sys
import os
import multiprocessing
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QLabel,
    QSlider,
    QHBoxLayout,
    QFormLayout,
    QDialog,
    QComboBox,
    QCheckBox,
)
from PyQt5.QtCore import Qt
import open3d as o3d
import numpy as np
import laspy
from basal_points_algo import (
    BasalPointSelection,
    BasalPointAlgorithm,
)
from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import os
import time
import shutil  # Import shutil for file operations
from geomdl import BSpline
from geomdl import utilities
import logging
import traceback
import csv
import pandas as pd
from pathlib import Path
from visualization import PointCloudVisualization
from point_cloud_io import PointCloudFileHandler
from mesh_processor import MeshProcessor
from geometric_analyzer import GeometricAnalyzer
from gui_components import GUIComponents
from utils import sor_filter, noise_filter, filter_point_cloud
from database_manager import DatabaseManager

# Set up logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='gui_log.txt'
)

# Region growing segmentation
def region_growing(self, pcd, rock_seeds, pedestal_seeds):
    """
    Run the region growing segmentation on the point cloud using the selected seeds.
    """
    from RegionGrowing import RegionGrowingSegmentation

    # Convert seed indices to integers
    rock_seed_indices = list(map(int, rock_seeds))
    pedestal_seed_indices = list(map(int, pedestal_seeds))

    print(rock_seed_indices, '---------', pedestal_seed_indices)

    # Initialize the region growing segmentation with the selected seeds and basal points
    #print('---------', self.basal_points)
    self.segmenter = RegionGrowingSegmentation(
        pcd,
        downsample=True,
        smoothness_threshold=self.smoothness_threshold,
        distance_threshold=0.05,
        curvature_threshold=self.curvature_threshold,
        rock_seeds=rock_seed_indices,
        pedestal_seeds=pedestal_seed_indices,
        basal_points=np.asarray(pcd.points)[self.basal_points] if np.any(self.basal_points) else None,
        basal_proximity_threshold=self.basal_proximity_threshold,
        stepwise_visualize=False,
    )

    # Segment the point cloud and perform conditional label propagation
    segmented_pcd, labels = self.segmenter.segment()
    #if not np.any(self.basal_points):
    self.segmenter.conditional_label_propagation()

    # Assign a default label for unlabeled points
    labels[labels == -1] = 1

    # Color the segmented point cloud
    colored_pcd = self.segmenter.color_point_cloud()

    # Highlight proximity points if basal points are available
    # if np.any(self.basal_points):
    #     colored_pcd = self.segmenter.highlight_proximity_points(colored_pcd)

    return colored_pcd


# Main window class for the GUI
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Rock segmentation tool")
        
        # Add user selection
        self.current_user = None
        self.users = ['Deep Rodge', 'Zhiang Chen', 'Ramon Arrowsmith']  # Add your users here
        
        # Initialize state variables
        self.pcd = None
        self.process = None
        self._active_processes = set()  # Track active processes
        self.rock_seeds = None
        self.pedestal_seeds = None
        self.poc_points = []
        self.basal_point_selection = None
        self.seed_selection_vis = None
        self.basal_points = None
        self.point_pick_queue = None 
        self.close_picking_event = None
        self.epsg_code = None  # Add EPSG code storage
        
        # Add mesh reconstruction state tracking
        self.mesh_reconstruction_stage = 0  # 0: not started, 1: intermediate, 2: complete
        self.prepared_mesh_data = None  # Store intermediate mesh data
        self.noise_removal_history = []  # Track history of noise removal steps for undo functionality
        
        # Initialize threshold values
        self.smoothness_threshold = 0.9
        self.curvature_threshold = 0.1
        self.basal_proximity_threshold = 0.05
        
        # Initialize file paths
        self.segmented_pcd_file_path = "Segmented point cloud was not saved"
        self.mesh_path = "Reconstructed Mesh was not saved"
        
        # Create central widget and initialize GUI components
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.gui = GUIComponents(central_widget)
        
        # Initialize processors
        self.visualizer = PointCloudVisualization()
        self.file_handler = PointCloudFileHandler()
        self.mesh_processor = MeshProcessor()
        self.geometric_analyzer = GeometricAnalyzer()
        self.db_manager = DatabaseManager()
        
        # Connect GUI signals
        self._connect_signals()

        # Initialize the user selection screen first
        self.show_user_selection()

    def show_user_selection(self):
        """Show the user selection screen."""
        # Create central widget for user selection
        user_widget = QWidget()
        layout = QVBoxLayout()
        
        # Add instructions
        instructions = QLabel("Please select your name from the list:")
        layout.addWidget(instructions)
        
        # Create user dropdown
        self.user_combo = QComboBox()
        self.user_combo.addItems(self.users)
        layout.addWidget(self.user_combo)
        
        # Add next button
        next_button = QPushButton("Next")
        next_button.clicked.connect(self.handle_user_selection)
        layout.addWidget(next_button)
        
        user_widget.setLayout(layout)
        self.setCentralWidget(user_widget)

    def handle_user_selection(self):
        """Handle the user selection and show the main interface."""
        self.current_user = self.user_combo.currentText()
        
        # Create central widget for main interface
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.gui = GUIComponents(central_widget)
        
        # Initialize processors
        self.visualizer = PointCloudVisualization()
        self.file_handler = PointCloudFileHandler()
        self.mesh_processor = MeshProcessor()
        self.geometric_analyzer = GeometricAnalyzer()
        self.db_manager = DatabaseManager()
        
        # Connect GUI signals
        self._connect_signals()
        
        # Show initial buttons
        self.gui.show_buttons(['load_database_button', 'load_button'])

    def start_visualization_process(self, target, args):
        """Start a visualization process and track it."""
        if self.process:
            self.process.terminate()
            self.process.join(timeout=1.0)
            if self.process in self._active_processes:
                self._active_processes.remove(self.process)
        
        self.process = multiprocessing.Process(target=target, args=args)
        self._active_processes.add(self.process)
        self.process.start()

    def _connect_signals(self):
        """Connect all GUI signals to their handlers."""
        signal_mappings = {
            'load_database_button': self.load_database,
            'load_button': lambda: self.load_las_file(None),  # Change this line to use lambda
            'next_pbr_button': self.load_next_pbr,
            'continue_button': self.continue_to_select_seeds,
            'false_positive_button': self.mark_as_false_positive,
            'manual_selection_button': self.start_manual_selection,
            'continue_with_seeds_button': lambda: self.show_sliders(),
            'next_button': self.select_pedestal_seeds,
            'run_button': self.run_region_growing,
            'basal_estimation_button': self.show_basal_estimation_options,
            'single_basal_button': self.input_point_of_contacts,
            'multi_basal_button': self.start_multi_part_basal_input,
            'estimate_basal_points_button': self.estimate_basal_points,
            'add_more_basal_points_button': self.add_more_basal_points,
            'rerun_region_growing_button': self.show_parameter_adjustment_window,
            'save_pcd_button': self.save_point_cloud,
            'reconstruct_mesh_button': self.reconstruct_mesh,
            'remove_noise_button': self.remove_noise_iterative,
            'undo_remove_noise_button': self.undo_remove_noise,  # Add new signal mapping
            'save_mesh_button': self.save_mesh,
            'compute_geometric_analysis_button': self.perform_geometric_analysis,
            'jump_to_basal_button': self.show_basal_estimation_options,
            'restart_button': self.restart_application,
        }
        
        for button_name, handler in signal_mappings.items():
            self.gui.connect_button(button_name, handler)
            
        # Connect slider value changed signals directly to the slider widgets
        self.gui.smoothness_slider.valueChanged.connect(self.update_smoothness_threshold)
        self.gui.curvature_slider.valueChanged.connect(self.update_curvature_threshold)
        self.gui.proximity_slider.valueChanged.connect(self.update_basal_proximity_threshold)

    def update_smoothness_threshold(self, value):
        self.smoothness_threshold = value / 100.0

    def update_curvature_threshold(self, value):
        self.curvature_threshold = value / 100.0

    def update_basal_proximity_threshold(self, value):
        self.basal_proximity_threshold = value / 100.0

    def remove_noise_iterative(self):
        """Remove noise iteratively from the prepared mesh data."""
        if self.mesh_reconstruction_stage == 1 and self.prepared_mesh_data:
            try:
                self.gui.set_instructions("Removing additional noise...")
                
                # Store current state in history before making changes
                self.noise_removal_history.append(self.prepared_mesh_data)
                
                # Unpack the prepared data
                rock_pcd, bottom_pcd, _, _ = self.prepared_mesh_data
                
                # Apply SOR filter to rock points again
                logging.info("Applying additional SOR filter to rock points...")
                rock_pcd_filtered, inlier_indices = rock_pcd.remove_statistical_outlier(
                    nb_neighbors=100, 
                    std_ratio=2.0
                )
                
                # Update the prepared data with filtered rock points
                combined_points_updated = np.vstack((
                    np.asarray(rock_pcd_filtered.points), 
                    np.asarray(bottom_pcd.points)
                ))
                combined_colors_updated = np.vstack((
                    np.full((len(np.asarray(rock_pcd_filtered.points)), 3), [1.0, 0.0, 0.0]),  # Red for rock points
                    np.full((len(np.asarray(bottom_pcd.points)), 3), [0.0, 1.0, 0.0])  # Green for bottom face points
                ))
                
                # Update prepared mesh data
                self.prepared_mesh_data = (rock_pcd_filtered, bottom_pcd, combined_points_updated, combined_colors_updated)
                
                # Update visualization
                self.start_visualization_process(
                    target=self.visualizer.show_point_cloud,
                    args=(combined_points_updated, combined_colors_updated, "After Additional Noise Removal")
                )
                
                # Show the buttons again (now including undo button)
                self.gui.set_instructions("New bottom face generated, and noise removed.\nPlease use below \"Remove Noise\" button to further remove the noise\nor \"Undo Remove Noise\" to revert last step\nor else \"Reconstruct Mesh\" to complete reconstruction.")
                self.gui.show_buttons(['reconstruct_mesh_button', 'remove_noise_button', 'undo_remove_noise_button'])
                
                logging.info(f"Additional noise removal completed. Kept {len(rock_pcd_filtered.points)} rock points.")
                
            except Exception as e:
                logging.error(f"Error in iterative noise removal: {str(e)}")
                self.gui.set_instructions("Error removing noise. Please try again.")

    def undo_remove_noise(self):
        """Undo the last noise removal step."""
        if self.mesh_reconstruction_stage == 1 and self.noise_removal_history:
            try:
                self.gui.set_instructions("Undoing last noise removal step...")
                
                # Restore the previous state from history
                self.prepared_mesh_data = self.noise_removal_history.pop()
                
                # Unpack the restored data for visualization
                rock_pcd, bottom_pcd, combined_points, combined_colors = self.prepared_mesh_data
                
                # Update visualization with restored data
                self.start_visualization_process(
                    target=self.visualizer.show_point_cloud,
                    args=(combined_points, combined_colors, "After Undo Noise Removal")
                )
                
                # Update button visibility based on remaining history
                buttons_to_show = ['reconstruct_mesh_button', 'remove_noise_button']
                if self.noise_removal_history:  # Only show undo button if there's history to undo
                    buttons_to_show.append('undo_remove_noise_button')
                
                self.gui.set_instructions("New bottom face generated, and noise removed.\nPlease use below \"Remove Noise\" button to further remove the noise" + 
                                         ("\nor \"Undo Remove Noise\" to revert last step" if self.noise_removal_history else "") +
                                         "\nor else \"Reconstruct Mesh\" to complete reconstruction.")
                self.gui.show_buttons(buttons_to_show)
                
                logging.info(f"Undo noise removal completed. Restored {len(np.asarray(rock_pcd.points))} rock points.")
                
            except Exception as e:
                logging.error(f"Error in undo noise removal: {str(e)}")
                self.gui.set_instructions("Error undoing noise removal. Please try again.")
        else:
            logging.warning("No noise removal history to undo")
            self.gui.set_instructions("No noise removal steps to undo.")

    def show_sliders(self):
        """Show parameter adjustment sliders."""
        self.gui.hide_all_buttons()
        has_basal_points = self.basal_points is not None and len(self.basal_points) > 0
        # Show sliders with current threshold values

        self.gui.show_buttons(['run_button', 'jump_to_basal_button'])
        self.gui.show_sliders(
            smoothness_threshold=self.smoothness_threshold,
            curvature_threshold=self.curvature_threshold,
            has_basal_points=has_basal_points
        )

    def load_las_file(self, file_name=None):
        """Load and display a LAS/LAZ file."""
        logging.debug(f"Load las file called with filename: {file_name}")
        
        if file_name is None:
            logging.debug("Opening file dialog")
            dialog = QFileDialog()
            dialog.setWindowTitle("Open LAS/LAZ File")
            dialog.setNameFilter("Point Cloud Files (*.las *.laz);;LAS Files (*.las);;LAZ Files (*.laz);;All Files (*)")
            dialog.setFileMode(QFileDialog.ExistingFile)
            
            if dialog.exec_() == QFileDialog.Accepted:
                file_name = dialog.selectedFiles()[0]
                logging.debug(f"Selected file: {file_name}")
            else:
                logging.debug("File dialog cancelled")
                return
        
        if file_name:
            try:
                self.input_path = Path(file_name)
                self.current_pbr_file = Path(file_name).stem
                self.output_folder = self.input_path.parent
                self.gui.set_instructions("")
                
                # Load the point cloud
                self.pcd, _, self.epsg_code = self.file_handler.load_las_as_open3d_point_cloud(file_name)
                
                # Log EPSG code
                if self.epsg_code is not None:
                    logging.info(f"Loaded point cloud with EPSG code: {self.epsg_code}")
                else:
                    logging.warning("No EPSG code found in point cloud file")
                
                # Visualize the point cloud using the new method
                self.start_visualization_process(
                    target=self.visualizer.show_point_cloud,
                    args=(np.asarray(self.pcd.points), np.asarray(self.pcd.colors), self.current_pbr_file)
                )
                
                # Show relevant buttons
                self.gui.show_buttons(['load_button', 'false_positive_button', 'continue_button'])
            except Exception as e:
                logging.error(f"Error loading file {file_name}: {str(e)}")
                self.gui.set_instructions(f"Error loading file: {str(e)}")

    def mark_as_false_positive(self):
        """Override to update database when marking false positive."""
        if hasattr(self, 'current_pbr_file'):
            self.db_manager.mark_false_positive(self.current_pbr_file)
            # Ensure we have input file information
            if not hasattr(self, 'input_path') or not hasattr(self, 'current_pbr_file'):
                self.gui.set_instructions("Error: No file is currently loaded")
                return
            
            # Create the false positives directory if it doesn't exist
            false_positives_dir = os.path.join(self.output_folder, "false_positives")
            os.makedirs(false_positives_dir, exist_ok=True)
            
            try:
                # Get the source file path and determine target path
                source_file = str(self.input_path)
                file_extension = os.path.splitext(source_file)[1]
                target_file = os.path.join(false_positives_dir, f"{self.current_pbr_file}{file_extension}")
                
                # Move the file to false positives directory
                shutil.move(source_file, target_file)
                
                # Add entry to the CSV file
                csv_path = os.path.join(self.output_folder, "false_positives.csv")
                with open(csv_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([self.current_pbr_file])

                self.gui.set_instructions(f"Marked {self.current_pbr_file} as false positive.\nFile moved to {false_positives_dir}")
            except Exception as e:
                logging.error(f"Error moving false positive file: {str(e)}", exc_info=True)
                self.gui.set_instructions(f"Error moving file: {str(e)}\nAdded to CSV only.")
                
                # Still add to CSV even if file operation failed
                csv_path = os.path.join(self.output_folder, "false_positives.csv")
                with open(csv_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([self.current_pbr_file])
            
            # Show load button
            self.gui.show_buttons(['load_button'])
            
            # Terminate any running visualization process
            if self.process:
                self.process.terminate()
            self.load_next_pbr()

    def continue_to_select_seeds(self):
        """Continue to seed selection after loading the point cloud."""
        # Apply filters
        self.gui.set_instructions("Preprocessing point cloud...")
        filtered_pcd, inlier_indices = filter_point_cloud(
            self.pcd,
            filter_type='sor',
            use_vertical_filter=True,  # Enable vertical filtering by default
            k_neighbors=10,
            std_ratio=2.0,
            vertical_std=1.0  # Use 1 standard deviation for vertical filter
        )
        self.pcd = filtered_pcd  # Update the point cloud
        
        # Computer selected seeds for rock and pedestal on filtered cloud
        points = np.asarray(self.pcd.points)
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)
        centroid_x = (min_bound[0] + max_bound[0]) / 2.0
        centroid_y = (min_bound[1] + max_bound[1]) / 2.0

        distances = np.linalg.norm(points[:, :2] - np.array([centroid_x, centroid_y]), axis=1)
        highest_point_index = np.argmax(points[:, 2] - distances)  # seed for rock
        bottommost_point_index = np.argmin(points[:, 2])  # seed for pedestal

        self.rock_seeds = [highest_point_index]
        self.pedestal_seeds = [bottommost_point_index]

        colors = np.full(points.shape, [0.5, 0.5, 0.5])
        self.pcd.colors = o3d.utility.Vector3dVector(colors)

        # Create seed points visualization
        seed_points = [
            (points[highest_point_index], [1, 0, 0]),     # Red sphere for rock
            (points[bottommost_point_index], [0, 0, 1])   # Blue sphere for pedestal
        ]

        self.start_visualization_process(
            target=self.visualizer.show_point_cloud, 
            args=(points, colors, self.current_pbr_file, False, seed_points)
        )

        # Show relevant buttons
        self.gui.show_buttons(['manual_selection_button', 'continue_with_seeds_button'])

    def pick_points(self, pcd):
        """Pick points from the point cloud using a separate process."""
        # Terminate any existing visualizations
        if self.process:
            self.process.terminate()
            self.process.join(timeout=1.0)
            if self.process in self._active_processes:
                self._active_processes.remove(self.process)

        # Create communication channels
        self.point_pick_queue = multiprocessing.Queue()
        self.close_picking_event = multiprocessing.Event()

        # Start point picking in new process
        self.start_visualization_process(
            target=self.visualizer.show_point_cloud_picking,
            args=(
                np.asarray(pcd.points), 
                np.asarray(pcd.colors), 
                self.point_pick_queue,
                self.close_picking_event,
                self.current_pbr_file
            )
        )

    def get_selected_points_close_window(self):
        """
        Retrieve selected points from the point picking process.
        """
        selected_points = []
        try:
            # Set the close event to trigger visualization closure
            if self.close_picking_event:
                self.close_picking_event.set()
                
            # Wait for points with timeout
            selected_points = self.point_pick_queue.get(timeout=5.0)
            print("Selected points:", selected_points)
            
        except Exception as e:
            print(f"Warning: Error while getting picked points: {e}")
        finally:
            # Clean up
            if self.process:
                self.process.terminate()
                self.process.join(timeout=1.0)
                self.process = None
            self.close_picking_event = None
        
        return selected_points
    
    def start_manual_selection(self):
        """Start manual selection of seeds."""
        self.gui.set_instructions(
            "Currently selecting seeds for Rock.\n \n"
            "1) Please pick points using [shift + left click].\n"
            "2) Press [shift + right click] to undo point picking.\n"
            "3) After picking points, press 'Next' to select Pedestal seeds."
        )
        
        # Disconnect any existing connections to avoid multiple triggers
        try:
            self.gui.next_button.clicked.disconnect()
        except Exception:
            pass
            
        # Explicitly connect to select_pedestal_seeds
        self.gui.connect_button('next_button', self.select_pedestal_seeds)
        self.gui.show_buttons(['next_button'])
        self.pick_points(self.pcd)

    def select_pedestal_seeds(self):
        """Select pedestal seeds after rock seeds have been selected."""
        self.rock_seeds = self.get_selected_points_close_window()
        
        if self.rock_seeds:
            self.gui.set_instructions(
                "Currently selecting seeds for Pedestal.\n \n"
                "1) Please pick points using [shift + left click].\n"
                "2) Press [shift + right click] to undo point picking.\n"
                "3) After picking points, press 'Next' to continue."
            )
            self.pick_points(self.pcd)
            
            # Disconnect any existing connections to avoid multiple triggers
            try:
                self.gui.next_button.clicked.disconnect()
            except Exception:
                pass
                
            # Explicitly connect to show_options_page
            self.gui.connect_button('next_button', self.show_options_page)
        else:
            self.gui.set_instructions("No rock seeds were selected. Please try selecting rock seeds again.")
            self.start_manual_selection()

    def show_options_page(self):
        """Show the options page after pedestal seed selection."""
        self.pedestal_seeds = self.get_selected_points_close_window()
        
        if self.pedestal_seeds:
            # Show sliders for parameter adjustment
            self.gui.show_sliders(
                smoothness_threshold=self.smoothness_threshold,
                curvature_threshold=self.curvature_threshold,
            )
            self.gui.show_buttons(['run_button', 'basal_estimation_button'])
            
            # # Add checkbox for DBSCAN cleaning
            # central_layout = self.centralWidget().layout()
            # if not hasattr(self, 'dbscan_checkbox'):
            #     self.dbscan_checkbox = QCheckBox("Clean point cloud using DBSCAN before mesh reconstruction")
            #     self.dbscan_checkbox.setChecked(False)  # Default to enabled
            #     central_layout.addWidget(self.dbscan_checkbox)
            # else:
            #     self.dbscan_checkbox.setVisible(True)
            
            self.gui.set_instructions(
                "Seeds selected successfully!\n\n"
                "You can now:\n"
                "1. Adjust parameters and run Region Growing\n"
                "2. Or jump directly to Basal Line selection"
            )
        else:
            self.gui.set_instructions("No pedestal seeds were selected. Please try selecting pedestal seeds again.")
            self.select_pedestal_seeds()

    def run_region_growing(self):
        """Run the region growing algorithm."""
        self.gui.hide_all_buttons()
        self.gui.set_instructions("Running Region Growing. Please wait...")

        colored_pcd = region_growing(self, self.pcd, self.rock_seeds, self.pedestal_seeds)
        self.gui.set_instructions("")

        # Show appropriate buttons based on state
        buttons_to_show = ['save_pcd_button', 'reconstruct_mesh_button', 'rerun_region_growing_button']
        if not np.any(self.basal_points) and not (hasattr(self, 'dense_basal_parts') and self.dense_basal_parts):
            buttons_to_show.append('basal_estimation_button')
        self.gui.show_buttons(buttons_to_show)

        self.start_visualization_process(
            target=self.visualizer.show_point_cloud,
            args=(np.asarray(colored_pcd.points), np.asarray(colored_pcd.colors), self.current_pbr_file)
        )

    def input_point_of_contacts(self):
        """
        Input points of contact for the region growing algorithm.
        """
        # Close all the visualization windows before starting point picking
        if self.process:
            self.process.terminate()

        self.gui.hide_all_buttons()
        self.gui.set_instructions(
            "Selecting Points of Contact.\n \n"
            "1) Please pick points using [shift + left click].\n"
            "2) Press [shift + right click] to undo point picking.\n"
            "3) After picking points, press 'Next' to estimate basal points."
        )
        
        # Disconnect any existing connections to avoid multiple triggers
        try:
            self.gui.next_button.clicked.disconnect()
        except:
            pass
        
        # Connect the next button to process the selected points
        self.gui.next_button.clicked.connect(self.process_single_basal_input)
        self.gui.show_buttons(['next_button'])

        # Call the function to allow the user to pick points
        self.basal_points = None
        self.pick_points(self.pcd)

    def process_single_basal_input(self):
        """Process the points selected for single basal input"""
        try:
            selected_points = self.get_selected_points_close_window()
            if selected_points is not None and len(selected_points) > 0:
                # Convert to list to ensure order is maintained
                selected_points = list(selected_points)
                # Create a temporary point cloud
                temp_pcd = o3d.geometry.PointCloud()
                temp_pcd.points = self.pcd.points
                
                # Run basal point estimation
                algorithm = BasalPointAlgorithm(temp_pcd)
                dense_points = algorithm.run(selected_points, show_progress=False, close_loop=True)
                
                if dense_points is not None and len(dense_points) > 0:
                    self.poc_points = dense_points
                    # Find indices for the dense points
                    tree = cKDTree(np.asarray(self.pcd.points))
                    _, self.basal_points = tree.query(dense_points)
                    
                    # Update visualization
                    points = np.asarray(self.pcd.points)
                    base_colors = np.full((len(points), 3), [0.5, 0.5, 0.5])  # Grey background
                    base_colors[self.basal_points] = [1, 0, 0]  # Red for basal points
                    
                    # Update point cloud colors
                    self.pcd.colors = o3d.utility.Vector3dVector(base_colors)
                    
                    self.start_visualization_process(
                        target=self.visualizer.show_point_cloud,
                        args=(points, base_colors, self.current_pbr_file)
                    )
                    
                    # Show parameter adjustment window
                    self.show_parameter_adjustment_window()
                else:
                    self.gui.set_instructions("Failed to generate dense points. Please try again.")
                    self.show_basal_estimation_options()
            else:
                self.gui.set_instructions("No points were selected. Please try again.")
                self.show_basal_estimation_options()
                
        except Exception as e:
            logging.error(f"Error in process_single_basal_input: {str(e)}")
            self.gui.set_instructions("An error occurred. Please try again.")
            self.show_basal_estimation_options()

    def add_more_basal_points(self):
        """
        Allow the user to reselect or add more basal points.
        """
        # Hide the buttons related to basal points and region growing
        self.gui.hide_buttons(['add_more_basal_points_button', 'run_button'])

        # Allow the user to reselect points of contact
        self.input_point_of_contacts()

    def estimate_basal_points(self):
        """Estimate basal points based on the selected points of contact."""
        logging.info("Starting basal point estimation")
        try:
            if not hasattr(self, 'poc_points') or len(self.poc_points) == 0:
                logging.warning("No points of contact available")
                raise ValueError("No points of contact available")

            self.gui.set_instructions("Estimating basal points. Please wait...")
            QApplication.processEvents()

            # Use BasalPointAlgorithm to estimate the basal points

            algorithm = BasalPointAlgorithm(self.pcd)

            # Handle multi-part basal points differently
            if hasattr(self, 'basal_parts') and len(self.basal_parts) > 1:
                logging.info(f"Processing {len(self.basal_parts)} parts for basal points")
                
                # Initialize storage for dense basal points of each part
                self.dense_basal_parts = []
                all_dense_points = []
                colors = []
                
                # Generate distinct colors for each part
                part_colors = [
                    [1, 0, 0],    # Red
                    [0, 1, 0],    # Green
                    [0, 0, 1],    # Blue
                    [1, 1, 0],    # Yellow
                    [1, 0, 1],    # Magenta
                ]

                points = np.asarray(self.pcd.points)
                last_point = None
                for i, part_indices in enumerate(self.basal_parts):
                    # Convert indices to integers and get the points
                    part_indices = np.array(part_indices, dtype=np.int32)
                    part_points = points[part_indices]
                    
                    if last_point is not None:
                        # Include the last point from previous part
                        part_points = np.vstack([last_point, part_points])
                    
                    # Store the last point of current part
                    last_point = part_points[-1]
                    
                    # Generate dense points for this part
                    dense_points = algorithm.run(part_points, show_progress=False)
                    
                    if dense_points is not None and len(dense_points) > 0:
                        self.dense_basal_parts.append(dense_points)
                        all_dense_points.extend(dense_points)
                        # Add colors for all points in this part
                        colors.extend([part_colors[i % len(part_colors)]] * len(dense_points))
                        logging.info(f"Generated {len(dense_points)} dense points for part {i+1}")
                    else:
                        logging.warning(f"No dense points generated for part {i+1}")

                # Convert to numpy arrays
                all_dense_points = np.array(all_dense_points)
                colors = np.array(colors)
                
                # Find indices of dense points in the point cloud
                tree = cKDTree(points)
                _, self.basal_points = tree.query(all_dense_points)

            else:
                # Original single-part processing
                dense_points = algorithm.run(self.poc_points, show_progress=False)
                if dense_points is None or len(dense_points) == 0:
                    logging.warning("No basal points were estimated")
                    raise ValueError("No basal points were estimated")
                    
                tree = cKDTree(np.asarray(self.pcd.points))
                _, self.basal_points = tree.query(dense_points)
                colors = np.array([[1, 0, 0]] * len(dense_points))  # All red for single part

            # Visualize the results
            points = np.asarray(self.pcd.points)
            base_colors = np.full((len(points), 3), [0.5, 0.5, 0.5])  # Grey background
            base_colors[self.basal_points] = colors  # Assign part colors to basal points
            
            # Update point cloud colors
            self.pcd.colors = o3d.utility.Vector3dVector(base_colors)

            # Show the visualization
            if self.process:
                self.process.terminate()
            
            self.process = multiprocessing.Process(
                target=self.visualizer.show_point_cloud,
                args=(points, base_colors, self.current_pbr_file)
            )
            self.process.start()

            # Show parameter adjustment window
            self.show_parameter_adjustment_window()

        except Exception as e:
            logging.error(f"Error in estimate_basal_points: {str(e)}", exc_info=True)
            self.gui.set_instructions("Failed to estimate basal points. Please try again.")
            self.show_basal_estimation_options()

    def save_point_cloud(self, plain = False):
        """
        Saves the point cloud to a LAS file with color-coded labels.
        Colors: Blue (pedestal), Red (rock), Green (basal points)
        
        Args:
            plain (bool): If True, saves all points in red without classification colors
        """
        # First, terminate any existing visualization process
        if self.process:
            self.process.terminate()
            self.process.join(timeout=1.0)
            self.process = None

        options = QFileDialog.Options()
        
        # Create default file name with pbr name
        default_filename = f"{self.current_pbr_file}_segmented.las" if hasattr(self, 'current_pbr_file') else "segmented.las"
        default_path = str(Path(self.output_folder) / default_filename) if hasattr(self, 'output_folder') else default_filename
        
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Point Cloud",
            default_path,
            "LAS Files (*.las);;All Files (*)",
            options=options,
        )
        if file_name:
            self.segmented_pcd_file_path = self.file_handler.save_point_cloud(
                self.pcd, file_name, np.asarray(self.segmenter.labels), self.basal_points, plain)
            
            
            # Update the instructions label to show success
            self.gui.set_instructions(f"Successfully saved point cloud to {file_name}")
            QApplication.processEvents()

    @staticmethod
    def detect_basal_points_optimized(points, labels, k=30, threshold=0.35):
        """
        Detects basal points using neighborhood analysis.
        """
        tree = cKDTree(points)
        distances, indices = tree.query(points, k=k)
        
        neighborhood_labels = labels[indices]
        rock_ratios = np.sum(neighborhood_labels == 1, axis=1) / k
        
        basal_points = (threshold <= rock_ratios) & (rock_ratios <= (1 - threshold))
        return basal_points

    def reconstruct_mesh(self):
        """Reconstruct a 3D mesh from the segmented point cloud."""
        self.gui.hide_all_buttons()
        
        if self.mesh_reconstruction_stage == 0:
            # First stage: Show intermediate visualization
            if self.basal_points is None:
                # Use optimized method to detect basal points
                self.basal_points = self.detect_basal_points_optimized(
                    np.asarray(self.pcd.points), np.asarray(self.segmenter.labels)
                )

            try:
                # # Check if DBSCAN cleaning option exists and is checked
                # use_dbscan = hasattr(self, 'dbscan_checkbox') and self.dbscan_checkbox.isChecked()
                
                self.gui.set_instructions("Preparing mesh reconstruction..." 
                                        #   +(" (with DBSCAN outlier removal)" if use_dbscan else "")
                                         )
                
                # Reset noise removal history when starting new reconstruction
                self.noise_removal_history = []
                
                # Get intermediate visualization data
                self.prepared_mesh_data = self.mesh_processor.reconstruct_mesh(
                    self.pcd,
                    labels=np.asarray(self.segmenter.labels),
                    basal_points=self.basal_points,
                    dense_basal_parts=self.dense_basal_parts if hasattr(self, 'dense_basal_parts') else None,
                    dense_basal_parts_is_lateral=self.dense_basal_parts_is_lateral if hasattr(self, 'dense_basal_parts_is_lateral') else None,
                    # use_dbscan_cleaning=use_dbscan,
                    debug_mode=False,  # No debug visualizations in GUI
                    intermediate_visualization=True  # Show intermediate and return early
                )
                
                # Handle visualization using separate process
                if self.prepared_mesh_data and len(self.prepared_mesh_data) == 4:
                    rock_pcd, bottom_pcd, combined_points, combined_colors = self.prepared_mesh_data
                    
                    # Start visualization in separate process
                    self.start_visualization_process(
                        target=self.visualizer.show_point_cloud,
                        args=(combined_points, combined_colors, "After SOR Filter")
                    )
                
                self.mesh_reconstruction_stage = 1
                self.gui.set_instructions("New bottom face generated, and noise removed.\nPlease use below \"Remove Noise\" button to further remove the noise\nor else \"Reconstruct Mesh\" to complete reconstruction.")
                self.gui.show_buttons(['remove_noise_button', 'reconstruct_mesh_button'])

            except Exception as e:
                logging.error(f"Error in mesh reconstruction preparation: {str(e)}")
                self.gui.set_instructions("Error in mesh preparation. Please try again.")
                self.mesh_reconstruction_stage = 0
                
        elif self.mesh_reconstruction_stage == 1:
            # Second stage: Complete the reconstruction
            try:
                self.gui.set_instructions("Completing mesh reconstruction...")
                
                # Unpack the prepared data
                rock_pcd, bottom_pcd, combined_points, combined_colors = self.prepared_mesh_data
                
                # Continue with mesh reconstruction
                combined_normals = np.vstack((
                    np.asarray(rock_pcd.normals),
                    np.asarray(bottom_pcd.normals)
                ))

                # Create final point cloud
                new_pcd = o3d.geometry.PointCloud()
                new_pcd.points = o3d.utility.Vector3dVector(combined_points)
                new_pcd.normals = o3d.utility.Vector3dVector(combined_normals)
                new_pcd.paint_uniform_color([1, 0, 0])

                # Perform Poisson reconstruction
                logging.info("Using Open3D Poisson reconstruction...")
                self.reconstructed_mesh = self.mesh_processor.poisson_reconstruction(
                    new_pcd,
                    depth=8,
                    width=0.0,
                    scale=1.5,
                    linear_fit=False
                )
                
                # Post-process the mesh
                self.reconstructed_mesh.remove_degenerate_triangles()
                self.reconstructed_mesh.remove_duplicated_triangles()
                self.reconstructed_mesh.remove_duplicated_vertices()
                self.reconstructed_mesh.remove_non_manifold_edges()
                
                # Store the mesh in mesh_processor for saving
                self.mesh_processor.reconstructed_mesh = self.reconstructed_mesh
                
                # Save temporary mesh for visualization
                with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as temp_file:
                    self.mesh_processor.temp_mesh_path = temp_file.name
                o3d.io.write_triangle_mesh(self.mesh_processor.temp_mesh_path, self.reconstructed_mesh)

                self.mesh_reconstruction_stage = 2
                self.gui.set_instructions("Mesh reconstruction completed.")
                self.gui.show_buttons(['save_mesh_button', 'compute_geometric_analysis_button', 'restart_button'])

                self.start_visualization_process(
                    target=self.visualizer.show_point_cloud, 
                    args=(self.mesh_processor.temp_mesh_path, None, self.current_pbr_file, True)
                )

            except Exception as e:
                logging.error(f"Error completing mesh reconstruction: {str(e)}")
                self.gui.set_instructions("Error completing mesh reconstruction. Please try again.")
                self.mesh_reconstruction_stage = 0
                
    def save_mesh(self):
        """
        Saves the reconstructed mesh to a PLY file.
        """
        options = QFileDialog.Options()
        
        # Create default file name with pbr name
        default_filename = f"{self.current_pbr_file}_mesh.ply" if hasattr(self, 'current_pbr_file') else "mesh.ply"
        default_path = str(Path(self.output_folder) / default_filename) if hasattr(self, 'output_folder') else default_filename
        
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Mesh",
            default_path,
            "PLY Files (*.ply);;All Files (*)",
            options=options,
        )
        if file_name:
            self.mesh_path = self.mesh_processor.save_mesh(file_name)
            self.gui.set_instructions(f"Mesh saved to {file_name}")

    def closeEvent(self, event):
        """Handle application closure."""
        if self.process:
            self.process.terminate()
        super().closeEvent(event)

    def show_parameter_adjustment_window(self):
        """Show window with parameter sliders and relevant buttons for rerunning region growing."""
        
        # Show additional buttons
        if np.any(self.basal_points):
            self.gui.show_buttons(['add_more_basal_points_button', 'run_button'])
            self.gui.show_sliders(
            smoothness_threshold=self.smoothness_threshold,
            curvature_threshold=self.curvature_threshold,
            has_basal_points=True
        )
        
        self.gui.set_instructions(
            "Basal points estimation completed. Would you like to add more basal points or run region growing again?"
        )

    def show_basal_estimation_options(self):
        """Show options for basal line estimation"""
        logging.info("Showing basal estimation options")
        self.gui.hide_all_buttons()
        self.gui.set_instructions(
            "Please select how you want to input the basal line:\n\n"
            "1. Input Complete Basal Line - Use this when the rock has a single continuous contact surface\n"
            "2. Input Basal Line in Parts - Use this when the rock rests on multiple surfaces"
        )
        self.gui.show_buttons(['single_basal_button', 'multi_basal_button'])

    def start_multi_part_basal_input(self):
        """Start the process of inputting basal line in multiple parts"""
        logging.info("Starting multi-part basal input")
        self.gui.hide_all_buttons()
        
        try:
            # Create dropdown for number of parts
            self.parts_combo = QComboBox()
            self.parts_combo.addItems(['2', '3', '4', '5'])
            
            # Create layout for selection
            dialog = QDialog(self)
            dialog.setWindowTitle("Select Number of Parts")
            layout = QVBoxLayout()
            layout.addWidget(QLabel("How many separate contact surfaces?"))
            layout.addWidget(self.parts_combo)
            
            # Add confirm button
            confirm_button = QPushButton("Confirm")
            confirm_button.clicked.connect(lambda: self.start_multi_part_input(int(self.parts_combo.currentText()), dialog))
            layout.addWidget(confirm_button)
            
            dialog.setLayout(layout)
            dialog.exec_()
        except Exception as e:
            logging.error(f"Error in start_multi_part_basal_input: {str(e)}", exc_info=True)

    def start_multi_part_input(self, num_parts, dialog):
        """Begin the multi-part basal input process"""
        logging.info(f"Starting multi-part input with {num_parts} parts")
        try:
            dialog.close()
            self.num_basal_parts = num_parts
            self.current_part = 0
            self.basal_parts = []  # List to store points for each part
            self.part_is_lateral = []  # List to track which parts are lateral points
            self.continue_multi_part_input()
        except Exception as e:
            logging.error(f"Error in start_multi_part_input: {str(e)}", exc_info=True)

    def continue_multi_part_input(self):
        """Handle input for each part of the basal line"""
        logging.info(f"Continuing multi-part input for part {self.current_part + 1}")
        try:
            self.current_part += 1
            self.gui.hide_all_buttons()
            
            # Get the central widget's layout
            central_layout = self.centralWidget().layout()
            
            # Add checkbox for lateral points
            if not hasattr(self, 'lateral_checkbox'):
                self.lateral_checkbox = QCheckBox()
            
            # Update checkbox text for current part
            self.lateral_checkbox.setText(f"Part {self.current_part}: These are lateral points (not basal contact points)")
            self.lateral_checkbox.setChecked(False)  # Default to false (basal points)
            
            # Remove previous checkbox if it exists in layout
            if hasattr(self, 'lateral_checkbox') and self.lateral_checkbox.parent():
                self.lateral_checkbox.setParent(None)
            
            # Add checkbox to layout
            central_layout.addWidget(self.lateral_checkbox)
            
            # Create layout for the last part's loop closure checkbox
            if self.current_part == self.num_basal_parts:
                # Add checkbox for loop closure
                if not hasattr(self, 'close_loop_checkbox'):
                    self.close_loop_checkbox = QCheckBox()
                
                self.close_loop_checkbox.setText("Close the loop (connect last point to first point)")
                self.close_loop_checkbox.setChecked(True)  # Default to checked
                
                # Remove previous loop checkbox if it exists in layout
                if hasattr(self, 'close_loop_checkbox') and self.close_loop_checkbox.parent():
                    self.close_loop_checkbox.setParent(None)
                
                central_layout.addWidget(self.close_loop_checkbox)
            
            self.gui.set_instructions(
                f"Selecting Points for Part {self.current_part} of {self.num_basal_parts}\n\n"
                "1) Please pick points using [shift + left click]\n"
                "2) Press [shift + right click] to undo point picking\n"
                "3) Check the box below if these are lateral points (not basal contact points)\n"
                "4) After picking points, press 'Next' to continue"
            )
            
            # Create and show next button if it doesn't exist
            if not hasattr(self, 'next_button'):
                logging.info("Creating next button")
                self.gui.next_button = QPushButton("Next")
                central_layout.addWidget(self.gui.next_button)
            
            # Disconnect any existing connections to avoid multiple triggers
            try:
                self.gui.next_button.clicked.disconnect()
            except:
                pass
            
            self.gui.next_button.clicked.connect(self.process_current_part)
            self.gui.show_buttons(['next_button'])
            
            # Start point picking
            logging.info("Starting point picking")
            self.pick_points(self.pcd)
            
        except Exception as e:
            logging.error(f"Error in continue_multi_part_input: {str(e)}", exc_info=True)

    def process_current_part(self):
        """Process the points selected for current part and continue if needed"""
        logging.info(f"Processing part {self.current_part}")
        try:
            current_points = self.get_selected_points_close_window()
            logging.info(f"Got {len(current_points) if current_points is not None else 0} points")
            
            if current_points is not None and len(current_points) > 0:
                # Convert to list to ensure order is maintained
                current_points = list(current_points)
                self.basal_parts.append(current_points)
                
                # Record whether this part is lateral points
                is_lateral = self.lateral_checkbox.isChecked()
                self.part_is_lateral.append(is_lateral)
                
                print("current_points", current_points)
                print(f"Part {self.current_part} is lateral: {is_lateral}")
                logging.info(f"Added points to basal_parts. Total parts: {len(self.basal_parts)}")
                logging.info(f"Part {self.current_part} marked as lateral: {is_lateral}")
                
                # Clean up the lateral checkbox
                if hasattr(self, 'lateral_checkbox') and self.lateral_checkbox.parent():
                    self.lateral_checkbox.setParent(None)
                
                if self.current_part < self.num_basal_parts:
                    logging.info("Moving to next part")
                    self.continue_multi_part_input()
                else:
                    logging.info("All parts collected, combining points")
                    # Store the loop closure preference
                    self.close_loop = self.close_loop_checkbox.isChecked()
                    # Remove the checkbox as it's no longer needed
                    if hasattr(self, 'close_loop_checkbox') and self.close_loop_checkbox.parent():
                        self.close_loop_checkbox.setParent(None)
                    
                    # Process each part separately with loop closure set to False
                    self.dense_basal_parts = []
                    self.dense_basal_parts_is_lateral = []  # Track which dense parts are lateral
                    
                    # Define distinct colors for different parts
                    part_colors = [
                        [1, 0, 0],    # Red
                        [0, 1, 0],    # Green
                        [0, 0, 1],    # Blue
                        [1, 1, 0],    # Yellow
                        [1, 0, 1],    # Magenta
                    ]
                    
                    # Initialize base colors (grey background)
                    points = np.asarray(self.pcd.points)
                    base_colors = np.full((len(points), 3), [0.5, 0.5, 0.5])
                    all_basal_indices = []
                    
                    for i, (part_points, is_lateral) in enumerate(zip(self.basal_parts, self.part_is_lateral)):
                        # Create a temporary point cloud for this part
                        temp_pcd = o3d.geometry.PointCloud()
                        temp_pcd.points = self.pcd.points
                        
                        # Run basal point estimation for this part
                        algorithm = BasalPointAlgorithm(temp_pcd)
                        # Pass False to prevent individual parts from closing loops, and enable visualization
                        part_type = "lateral" if is_lateral else "basal"
                        self.gui.set_instructions(f"Processing {part_type} part {i+1}/{len(self.basal_parts)}...")
                        dense_part = algorithm.run(part_points, show_progress=False, close_loop=False)
                        if dense_part is not None and len(dense_part) > 0:
                            self.dense_basal_parts.append(dense_part)
                            self.dense_basal_parts_is_lateral.append(is_lateral)
                            
                            # Find indices for this part's points
                            tree = cKDTree(points)
                            _, part_indices = tree.query(dense_part)
                            all_basal_indices.extend(part_indices)
                            
                            # Color this part's points with its designated color
                            # Use lighter color for lateral points
                            color = part_colors[i % len(part_colors)]  # Cycle through colors if more parts than colors
                            if is_lateral:
                                # Make lateral points lighter (add 0.3 to each channel, capped at 1.0)
                                color = [min(1.0, c + 0.3) for c in color]
                            base_colors[part_indices] = color
                    
                    # Combine all parts for region growing
                    if self.dense_basal_parts:
                        self.poc_points = np.concatenate(self.dense_basal_parts)
                        self.basal_points = np.array(all_basal_indices)
                        logging.info(f"Combined {len(self.poc_points)} total points")
                        
                        # Update point cloud colors
                        self.pcd.colors = o3d.utility.Vector3dVector(base_colors)
                        
                        # Show the visualization
                        if self.process:
                            self.process.terminate()
                        
                        self.process = multiprocessing.Process(
                            target= self.visualizer.show_point_cloud,
                            args=(points, base_colors, self.current_pbr_file)
                        )
                        self.process.start()
                        
                        # Show parameter adjustment window
                        self.show_parameter_adjustment_window()
                    else:
                        logging.warning("No dense parts were generated")
                        self.gui.set_instructions("Failed to generate dense points. Please try again.")
                        self.show_basal_estimation_options()
            else:
                logging.warning("No points were selected")
                self.gui.set_instructions("No points were selected. Please try again.")
                self.continue_multi_part_input()
                
        except Exception as e:
            logging.error(f"Error in process_current_part: {str(e)}", exc_info=True)
            self.gui.set_instructions("An error occurred. Please try again.")
            self.show_basal_estimation_options()

    def perform_geometric_analysis(self):
        """
        Perform geometric analysis on the reconstructed mesh.
        """
        self.gui.hide_all_buttons()
        self.gui.set_instructions("Computing geometric properties...")

        try:
            # Get pedestal points if available
            pedestal_points = None
            if hasattr(self, 'segmenter'):
                pedestal_mask = self.segmenter.labels == 0
                pedestal_points = np.asarray(self.pcd.points)[pedestal_mask]

            # Prepare basal points and lateral flags for geometric analysis
            basal_points_coords = np.asarray(self.pcd.points)[self.basal_points]
            
            # Create lateral flags for each basal point
            lateral_flags = None
            if hasattr(self, 'dense_basal_parts_is_lateral') and hasattr(self, 'dense_basal_parts'):
                # Create a flag array for each basal point
                lateral_flags = []
                current_idx = 0
                
                for i, (dense_part, is_lateral) in enumerate(zip(self.dense_basal_parts, self.dense_basal_parts_is_lateral)):
                    part_size = len(dense_part)
                    # Add the lateral flag for all points in this part
                    lateral_flags.extend([is_lateral] * part_size)
                    current_idx += part_size
                
                lateral_flags = np.array(lateral_flags)
                logging.info(f"Created lateral flags: {len(lateral_flags)} flags, {sum(lateral_flags)} lateral points")

            # Compute geometric properties using the geometric analyzer
            results = self.geometric_analyzer.compute_geometric_properties(
                self.mesh_processor.reconstructed_mesh,
                basal_points_coords,
                pedestal_points,
                lateral_flags=lateral_flags  # Pass lateral flags to the analyzer
            )

            # Add user information to results
            results['user'] = self.current_user

            # Save results to CSV
            csv_path = self.geometric_analyzer.save_results(
                results,
                self.current_pbr_file,
                self.input_path,
                self.segmented_pcd_file_path,
                self.mesh_path,
                self.smoothness_threshold,
                self.curvature_threshold,
                self.basal_proximity_threshold,
                self.current_user,
                self.epsg_code
            )

            self.gui.set_instructions(
                f"Geometric analysis completed.\nResults saved to: {csv_path}\n\nShowing alpha angle view..."
            )

            # Update database with results
            self.db_manager.update_entry(
                self.current_pbr_file,
                processed=True,
                segmented_pbr_location=self.segmented_pcd_file_path,
                mesh_reconstruction_location=self.mesh_path,
                smoothness_threshold=self.smoothness_threshold,
                curvature_threshold=self.curvature_threshold,
                proximity_threshold=self.basal_proximity_threshold,
                **results
            )
            
            # Show alpha view visualization
            if hasattr(self.mesh_processor, 'temp_mesh_path') and self.mesh_processor.temp_mesh_path:
                self.start_visualization_process(
                    target=self.visualizer.show_mesh_with_alpha_view,
                    args=(
                        self.mesh_processor.temp_mesh_path,
                        results['center_of_mass'],
                        results['alpha_plane_normal'],
                        f"Alpha View - {self.current_pbr_file} (α={results['alpha_angle']:.1f}°)",
                        pedestal_points,  # Pass pedestal points
                        results['min_alpha_basal_point']  # Pass the alpha angle point
                    )
                )
            
            # Show next PBR button
            self.gui.show_buttons(['save_mesh_button', 'restart_button', 'next_pbr_button'])

        except Exception as e:
            self.gui.set_instructions(f"Error in geometric analysis: {str(e)}")
            logging.error(f"Error in geometric analysis: {str(e)}")
            
        finally:
            self.gui.show_buttons(['save_mesh_button', 'restart_button', 'next_pbr_button'])

    def restart_application(self):
        """Reset the application to the initial state."""
        # Terminate any running visualization process
        if self.process:
            self.process.terminate()
            self.process.join(timeout=1.0)
            self.process = None
        
        # Reset all state variables
        self.pcd = None
        self.rock_seeds = None
        self.pedestal_seeds = None
        self.poc_points = []
        self.basal_point_selection = None
        self.basal_points = None
        self.point_pick_queue = None
        self.close_picking_event = None
        
        # Reset any multipart basal related attributes
        if hasattr(self, 'dense_basal_parts'):
            delattr(self, 'dense_basal_parts')
        if hasattr(self, 'basal_parts'):
            delattr(self, 'basal_parts')
        
        # Reset the segmenter if it exists
        if hasattr(self, 'segmenter'):
            delattr(self, 'segmenter')
        
        # Reset threshold values to defaults
        self.smoothness_threshold = 0.9
        self.curvature_threshold = 0.1
        self.basal_proximity_threshold = 0.05
        
        # Reset file paths
        self.segmented_pcd_file_path = "Segmented point cloud was not saved"
        self.mesh_path = "Reconstructed Mesh was not saved"
        
        # Reset mesh reconstruction state and history
        self.mesh_reconstruction_stage = 0
        self.prepared_mesh_data = None
        self.noise_removal_history = []
        
        # Reset button connections to avoid signal conflicts
        try:
            if hasattr(self.gui, 'next_button'):
                self.gui.next_button.clicked.disconnect()
        except Exception:
            pass
            
        # Hide all buttons and show only the load button
        self.gui.hide_all_buttons()
        self.gui.show_buttons(['load_button'])
        self.gui.set_instructions("Application restarted. Please load a new point cloud.")

    def load_database(self):
        """Load database file and start processing PBRs."""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open Database File",
            "",
            "CSV Files (*.csv);;All Files (*)",
            options=options
        )
        
        if file_name:
            try:
                if self.db_manager.load_database(file_name):
                    self.load_next_pbr()
                else:
                    self.gui.set_instructions("Error loading database file")
            except Exception as e:
                logging.error(f"Database loading error: {str(e)}", exc_info=True)
                self.gui.set_instructions(f"Error loading database: {str(e)}")

    def load_next_pbr(self):
        """Load the next unprocessed PBR from database."""
        # Reset all state variables
        self.pcd = None
        self.rock_seeds = None
        self.pedestal_seeds = None
        self.poc_points = []
        self.basal_point_selection = None
        self.basal_points = None
        self.point_pick_queue = None
        self.close_picking_event = None
        
        # Reset any multipart basal related attributes
        if hasattr(self, 'dense_basal_parts'):
            delattr(self, 'dense_basal_parts')
        if hasattr(self, 'basal_parts'):
            delattr(self, 'basal_parts')
        
        # Reset the segmenter if it exists
        if hasattr(self, 'segmenter'):
            delattr(self, 'segmenter')
        
        # Reset threshold values to defaults
        self.smoothness_threshold = 0.9
        self.curvature_threshold = 0.1
        self.basal_proximity_threshold = 0.05
        
        # Reset file paths
        self.segmented_pcd_file_path = "Segmented point cloud was not saved"
        self.mesh_path = "Reconstructed Mesh was not saved"
        
        # Reset mesh reconstruction state and history
        self.mesh_reconstruction_stage = 0
        self.prepared_mesh_data = None
        self.noise_removal_history = []
        
        # Terminate any running visualization process
        if self.process:
            self.process.terminate()
            self.process.join(timeout=1.0)
            self.process = None
        
        try:
            next_pbr = self.db_manager.get_next_unprocessed()
            if next_pbr is not None:
                # Get the full path relative to the database directory
                full_file_path = self.db_manager.get_full_file_path(next_pbr['pbr_location'])
                self.load_las_file(full_file_path)
            else:
                self.gui.set_instructions("No more unprocessed PBRs in database")
        except Exception as e:
            logging.error(f"Error getting next PBR: {str(e)}", exc_info=True)
            self.gui.set_instructions(f"Error getting next PBR: {str(e)}")
# Main entry point for the application
if __name__ == "__main__":
    # Suppress Open3D and GLFW warnings
    import os
    import warnings
    
    # Set environment variable to reduce GLFW verbosity
    os.environ['OPEN3D_VERBOSE'] = '0'
    
    # For macOS: suppress common Cocoa display warnings
    if sys.platform == 'darwin':
        os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
        
    # Filter Open3D warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="open3d")
    
    # Configure Open3D verbosity level (set to 0 to disable most output)
    try:
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    except:
        pass
    
    # Suppress IMKClient and IMKInputSession messages by redirecting stderr temporarily
    # (These are macOS specific input method warnings that can't be easily silenced)
    
    # Set the start method for multiprocessing to 'spawn' (required for some platforms)
    multiprocessing.set_start_method("spawn")

    # Create the application instance and main window
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    # Start the application's event loop
    sys.exit(app.exec_())
