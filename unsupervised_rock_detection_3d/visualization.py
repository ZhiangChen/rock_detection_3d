import open3d as o3d
import numpy as np
import logging
import multiprocessing
from multiprocessing import Queue, Event
import tempfile
from pathlib import Path
from typing import Optional

class PointCloudVisualization:
    """
    Handles all visualization-related operations for point clouds and meshes using Open3D.
    """
    
    @staticmethod
    def show_point_cloud(points_or_mesh_path, colors=None,  window_name="Open3D Visualization", is_mesh=False, seed_points=None, point_show_normal=False, normals=None):
        """
        Visualize the point cloud or mesh using Open3D.
        
        Args:
            points_or_mesh_path: Either numpy array of points or path to mesh file
            colors: Optional numpy array of colors for the points
            is_mesh: Boolean indicating if input is a mesh path
            seed_points: Optional list of (point, color) tuples for seed visualization
            point_show_normal: Boolean to show point normals
            normals: Optional numpy array of normals for the points
        """
        geometries = []
        
        if not is_mesh:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_or_mesh_path)
            if colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(colors)
            if normals is not None:
                pcd.normals = o3d.utility.Vector3dVector(normals)
            else:
                # Estimate normals if not provided
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50)
                )
            geometries.append(pcd)
            
            # Add spheres for seed points if provided
            if seed_points is not None:
                for point, color in seed_points:
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
                    sphere.translate(point)
                    sphere.paint_uniform_color(color)
                    geometries.append(sphere)
        else:
            # Load the mesh from the file
            geometry = o3d.io.read_triangle_mesh(points_or_mesh_path)
            geometries.append(geometry)
            
        o3d.visualization.draw_geometries(geometries, window_name=window_name, point_show_normal=point_show_normal)


    @staticmethod
    def show_point_cloud_picking(points, colors, queue, close_event, window_name="Select Points"):
        """
        Show point cloud for picking points and send picked points through queue.
        
        Args:
            points: numpy array of points
            colors: numpy array of colors
            queue: multiprocessing Queue for returning picked points
            close_event: multiprocessing Event for signaling visualization closure
            window_name: Title of the visualization window
        """
        try:
            # Create visualization window
            vis = o3d.visualization.VisualizerWithEditing()
            vis.create_window(window_name=window_name)

            # Create point cloud and add to visualizer
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Estimate normals for better visualization
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50)
            )
            
            vis.add_geometry(pcd)
            
            # Create a list to store picked points in order
            picked_points = []
            last_picked = set()  # Keep track of last state to detect deselections
            
            def pick_points_callback(vis):
                nonlocal last_picked, picked_points
                if close_event.is_set():
                    # If close event is set, get current picked points and close window
                    queue.put(picked_points)
                    vis.close()
                    return True
                
                current_picked = set(vis.get_picked_points())
                
                # Handle deselections
                if len(current_picked) < len(last_picked):
                    # Find which point was deselected
                    deselected = last_picked - current_picked
                    # Remove the deselected point from our ordered list
                    for point_idx in deselected:
                        if point_idx in picked_points:
                            picked_points.remove(point_idx)
                
                # Handle new selections
                for point_idx in current_picked - last_picked:
                    if point_idx not in picked_points:
                        picked_points.append(point_idx)
                
                # Update last_picked for next comparison
                last_picked = current_picked
                return False
            
            # Register the callback
            vis.register_animation_callback(pick_points_callback)
            
            # Run the visualizer
            vis.run()
            
            # Clean up
            vis.destroy_window()
            
        except Exception as e:
            logging.error(f"Error in point picking visualization: {e}")
            queue.put([])  # Send empty list in case of error

    @staticmethod
    def highlight_points(pcd, points_array):
        """
        Highlight specific points in the point cloud by coloring them.
        
        Args:
            pcd: Open3D PointCloud object
            points_array: numpy array of points to highlight
            
        Returns:
            numpy array of colors with highlighted points
        """
        try:
            if points_array is None or len(points_array) == 0:
                logging.warning("No points to highlight")
                return np.asarray(pcd.colors)

            # Create a copy of the original colors
            colors = np.asarray(pcd.colors).copy()
            
            # Find the indices of points to highlight using nearest neighbor search
            tree = o3d.geometry.KDTreeFlann(pcd)
            for point in points_array:
                _, idx, _ = tree.search_knn_vector_3d(point, 1)
                colors[idx[0]] = [1, 0, 0]  # Red color for highlighted points
            
            return colors

        except Exception as e:
            logging.error(f"Error in highlight_points: {str(e)}")
            return np.asarray(pcd.colors)
    
    @staticmethod
    def show_mesh_with_alpha_view(mesh_path: str, center_of_mass: np.ndarray, 
                                alpha_plane_normal: np.ndarray, window_title: str = "Alpha View",
                                pedestal_points: Optional[np.ndarray] = None,
                                alpha_angle_point: Optional[np.ndarray] = None):
        """
        Show mesh with camera view perpendicular to the alpha plane.
        
        Args:
            mesh_path: Path to the mesh file
            center_of_mass: Center of mass of the mesh
            alpha_plane_normal: Normal vector to the alpha plane
            window_title: Title for the visualization window
            pedestal_points: Optional array of pedestal points to display
            alpha_angle_point: Optional point that produces the minimum alpha angle (to highlight)
        """
        try:
            import open3d as o3d
            import numpy as np
            
            # Load the mesh
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            if len(mesh.vertices) == 0:
                print(f"Warning: Could not load mesh from {mesh_path}")
                return
            
            # Create list of geometries to display
            geometries = [mesh]
            
            # Add pedestal points if provided
            if pedestal_points is not None and len(pedestal_points) > 0:
                pedestal_pcd = o3d.geometry.PointCloud()
                pedestal_pcd.points = o3d.utility.Vector3dVector(pedestal_points)
                # Color pedestal points blue
                pedestal_pcd.paint_uniform_color([0.0, 0.0, 1.0])  # Blue
                geometries.append(pedestal_pcd)
                print(f"Added {len(pedestal_points)} pedestal points (blue)")
            
            # Add alpha angle point as a highlighted sphere if provided
            if alpha_angle_point is not None:
                # Create a small sphere at the alpha angle point
                alpha_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
                alpha_sphere.translate(alpha_angle_point)
                # Color the sphere bright green to highlight it
                alpha_sphere.paint_uniform_color([0.0, 1.0, 0.0])  # Bright green
                geometries.append(alpha_sphere)
                print(f"Added alpha angle point sphere at {alpha_angle_point} (bright red)")
            
            # Calculate camera parameters for alpha view
            # The camera should look perpendicular to the alpha plane
            lookat = center_of_mass
            
            # Camera front vector is the alpha plane normal (perpendicular to plane)
            front = alpha_plane_normal / np.linalg.norm(alpha_plane_normal)
            
            # Calculate up vector - use global Z if possible, otherwise choose perpendicular vector
            z_axis = np.array([0, 0, 1])
            if abs(np.dot(front, z_axis)) < 0.9:  # Not parallel to Z
                up = z_axis - np.dot(z_axis, front) * front
                up = up / np.linalg.norm(up)
            else:
                # If front is parallel to Z, use Y axis
                y_axis = np.array([0, 1, 0])
                up = y_axis - np.dot(y_axis, front) * front
                up = up / np.linalg.norm(up)
            
            # Set zoom for good view
            zoom = 0.7
            
            print(f"Alpha view parameters:")
            print(f"  Lookat (center of mass): {lookat}")
            print(f"  Front (alpha plane normal): {front}")
            print(f"  Up: {up}")
            print(f"  Zoom: {zoom}")
            
            # Create visualization parameters
            vis_params = {
                "window_name": window_title,
                "width": 1280,
                "height": 720,
                "lookat": lookat,
                "up": up,
                "front": front,
                "zoom": zoom
            }
            
            # Show all geometries with alpha view
            o3d.visualization.draw_geometries(geometries, **vis_params)
            
        except Exception as e:
            print(f"Error in alpha view visualization: {e}")
            import traceback
            traceback.print_exc()