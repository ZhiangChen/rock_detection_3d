from PyQt5.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QWidget,
    QLabel,
    QSlider,
    QHBoxLayout,
    QFormLayout,
    QDialog,
)
from PyQt5.QtCore import Qt
import logging

class GUIComponents:
    """
    A class to manage all GUI components that can be reused across different windows.
    """
    
    def __init__(self, parent_widget: QWidget):
        """
        Initialize GUI components.
        
        Args:
            parent_widget: The parent widget to attach the components to
        """
        self.parent = parent_widget
        self.layout = QVBoxLayout()
        self.initialize_components()
        
    def initialize_components(self):
        """Initialize all GUI components but keep them hidden initially."""

        # Instructions label
        self.instructions_label = QLabel("")
        self.layout.addWidget(self.instructions_label)
                
        # Create sliders layout
        self.slider_layout = QFormLayout()
        self._setup_sliders()
        slider_widget = QWidget()
        slider_widget.setLayout(self.slider_layout)
        slider_widget.setVisible(False)
        self.slider_widget = slider_widget
        self.layout.addWidget(slider_widget)

        # File handling buttons
        self.load_button = self._create_button("Load LAS File")
        self.continue_button = self._create_button("Continue to Select Seeds", visible=False)
        self.false_positive_button = self._create_button("Mark as False Positive", visible=False)
        
        # Seed selection buttons
        self.manual_selection_button = self._create_button("Select Seeds Manually", visible=False)
        self.continue_with_seeds_button = self._create_button("Continue with Selected Seeds", visible=False)
        self.next_button = self._create_button("Next", visible=False)
        
        # Basal point buttons
        self.basal_estimation_button = self._create_button("Basal Line Estimation", visible=False)
        self.single_basal_button = self._create_button("Input Complete Basal Line", visible=False)
        self.multi_basal_button = self._create_button("Input Basal Line in Parts", visible=False)
        self.estimate_basal_points_button = self._create_button("Estimate Basal Points", visible=False)
        self.add_more_basal_points_button = self._create_button("Reselect Basal Points", visible=False)
        
        # Processing buttons
        self.run_button = self._create_button("Run Region Growing", visible=False)
        self.rerun_region_growing_button = self._create_button("Rerun Region Growing with Different Parameters", visible=False)
        self.save_pcd_button = self._create_button("Save Point Cloud", visible=False)
        self.reconstruct_mesh_button = self._create_button("Reconstruct Mesh", visible=False)
        self.save_mesh_button = self._create_button("Save Mesh", visible=False)
        self.compute_geometric_analysis_button = self._create_button("Compute Geometric Properties", visible=False)
        self.jump_to_basal_button = self._create_button("Jump to Basal Line Selection", visible=False)
        self.restart_button = self._create_button("Restart", visible=False)

        
        # Set the layout for the parent widget
        self.parent.setLayout(self.layout)
        
    def _create_button(self, text: str, visible: bool = True, enabled: bool = True) -> QPushButton:
        """Create a button with given properties and add it to the layout."""
        button = QPushButton(text)
        button.setVisible(visible)
        button.setEnabled(enabled)
        self.layout.addWidget(button)
        return button
        
    def _setup_sliders(self):
        """Set up the parameter adjustment sliders."""
        # Slider descriptions
        smoothness_description = QLabel("Controls surface smoothness variation; higher values include only smoother points.\n")
        curvature_description = QLabel("Sets the curvature limit; higher values allow more curved points.\n")
        
        # Smoothness Threshold Slider
        self.smoothness_container, self.smoothness_slider, smoothness_value_label = self._create_slider(0.99, "Smoothness Threshold")
        self.slider_layout.addRow("Smoothness Threshold", self.smoothness_container)
        self.slider_layout.addRow(smoothness_description)
        
        # Curvature Threshold Slider
        self.curvature_container, self.curvature_slider, curvature_value_label = self._create_slider(0.15, "Curvature Threshold")
        self.slider_layout.addRow("Curvature Threshold", self.curvature_container)
        self.slider_layout.addRow(curvature_description)
        
        # Basal Proximity Threshold Slider
        self.proximity_slider_label = QLabel("Basal Proximity Threshold")
        self.proximity_container, self.proximity_slider, self.proximity_value_label = self._create_slider(0.05, "Basal Proximity Threshold")
        self.slider_layout.addRow(self.proximity_slider_label, self.proximity_container)
        
    def _create_slider(self, initial_value: float, name: str) -> tuple:
        """Create a slider with given initial value and name."""
        slider_layout = QHBoxLayout()
        
        # Create the slider widget
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(int(initial_value * 100))
        slider.setMinimumWidth(300)
        
        # Create value label
        value_label = QLabel(f"{initial_value:.2f}")
        slider.valueChanged.connect(lambda v: value_label.setText(f"{v/100.0:.2f}"))
        
        # Add widgets to layout
        slider_layout.addWidget(slider)
        slider_layout.addWidget(value_label)
        
        # Create a container widget to hold the layout
        container = QWidget()
        container.setLayout(slider_layout)
        
        # Store both the slider and its container
        return container, slider, value_label
    
    def show_sliders(self, smoothness_threshold: float = 0.9, curvature_threshold: float = 0.1, 
                    has_basal_points: bool = False):
        """Show the parameter adjustment sliders."""
        self.slider_widget.setVisible(True)
        self.smoothness_slider.setValue(int(smoothness_threshold * 100))
        self.curvature_slider.setValue(int(curvature_threshold * 100))
        
        # Show proximity slider only if basal points are present
        self.proximity_slider_label.setVisible(has_basal_points)
        self.proximity_container.setVisible(has_basal_points)
        
    def hide_all_buttons(self):
        """Hide all buttons in the GUI."""
        for widget in self.parent.findChildren(QPushButton):
            widget.setVisible(False)
        self.slider_widget.setVisible(False)

    def hide_buttons(self, button_list: list):
        """Hide specified buttons in the GUI."""
        for button_name in button_list:
            if hasattr(self, button_name):
                getattr(self, button_name).setVisible(False)
        
    def show_buttons(self, button_list: list):
        """Show specified buttons and hide others."""
        self.hide_all_buttons()
        for button_name in button_list:
            if hasattr(self, button_name):
                getattr(self, button_name).setVisible(True)
                
    def set_instructions(self, text: str):
        """Set the text of the instructions label."""
        self.instructions_label.setText(text)
        
    def get_slider_values(self) -> dict:
        """Get the current values of all sliders."""
        return {
            'smoothness': self.smoothness_slider.value() / 100.0,
            'curvature': self.curvature_slider.value() / 100.0,
            'proximity': self.proximity_slider.value() / 100.0
        }
        
    def connect_button(self, button_name: str, callback):
        """Connect a button to its callback function."""
        if hasattr(self, button_name):
            getattr(self, button_name).clicked.connect(callback)
        else:
            logging.warning(f"Button {button_name} not found")