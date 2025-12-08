import logging
import multiprocessing
import sys
from multiprocessing import Queue, Event
from typing import Optional, Tuple

import numpy as np
import open3d as o3d

class PointCloudVisualization:
    """
    Handles all visualization-related operations for point clouds and meshes using Open3D.
    """

    @staticmethod
    def _get_visualizer_dimensions() -> Tuple[int, int, int, int]:
        """Return (width, height, left, top) for Open3D windows constrained to half screen width."""
        default_width, default_height = 960, 720
        left, top = 0, 0

        try:
            if sys.platform == "darwin":
                from AppKit import NSScreen  # type: ignore

                screen = NSScreen.mainScreen()
                if screen is not None:
                    frame = screen.visibleFrame()
                    width = max(640, int(frame.size.width / 2))
                    height = max(480, int(frame.size.height * 0.85))
                    # macOS origin is bottom-left; Open3D expects top-left for positioning.
                    default_width = width
                    default_height = height
                    left = int(frame.origin.x)
                    top = int(frame.origin.y + frame.size.height - height)
            elif sys.platform.startswith("win"):
                from ctypes import windll

                user32 = windll.user32
                user32.SetProcessDPIAware()
                screen_width = user32.GetSystemMetrics(0)
                screen_height = user32.GetSystemMetrics(1)
                default_width = max(640, screen_width // 2)
                default_height = max(480, int(screen_height * 0.85))
            else:
                import tkinter as tk

                root = tk.Tk()
                root.withdraw()
                screen_width = root.winfo_screenwidth()
                screen_height = root.winfo_screenheight()
                default_width = max(640, screen_width // 2)
                default_height = max(480, int(screen_height * 0.85))
                root.destroy()
        except Exception:
            logging.debug("Falling back to default Open3D window size for visualization.")

        return default_width, default_height, left, top
    
    @staticmethod
    def show_point_cloud(
        points_or_mesh_path,
        colors=None,
        window_name="Open3D Visualization",
        is_mesh=False,
        seed_points=None,
        point_show_normal=False,
        normals=None,
        show_wireframe=None,
    ):
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
        render_wireframe = is_mesh if show_wireframe is None else bool(show_wireframe)
        
        width, height, left, top = PointCloudVisualization._get_visualizer_dimensions()

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
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
                    sphere.translate(point)
                    sphere.paint_uniform_color(color)
                    sphere.compute_vertex_normals()
                    geometries.append(sphere)
        else:
            # Load the mesh from the file
            geometry = o3d.io.read_triangle_mesh(points_or_mesh_path)
            if len(geometry.vertices) == 0:
                logging.warning("Empty mesh encountered for visualization: %s", points_or_mesh_path)
                return

            geometry.compute_vertex_normals()
            geometries.append(geometry)

            if render_wireframe and len(geometry.triangles) > 0:
                wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(geometry)
                wireframe.paint_uniform_color([0.0, 0.0, 0.0])
                geometries.append(wireframe)
            
        o3d.visualization.draw_geometries(
            geometries,
            window_name=window_name,
            point_show_normal=point_show_normal,
            width=width,
            height=height,
            left=left,
            top=top,
        )


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
            width, height, left, top = PointCloudVisualization._get_visualizer_dimensions()
            # Create visualization window
            vis = o3d.visualization.VisualizerWithEditing()
            vis.create_window(window_name=window_name, width=width, height=height, left=left, top=top)

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
    def show_mesh_with_alpha_view(
        mesh_path: str,
        center_of_mass: np.ndarray,
        alpha_plane_normal: np.ndarray,
        window_title: str = "Alpha View",
        pedestal_points: Optional[np.ndarray] = None,
        alpha_angle_point: Optional[np.ndarray] = None,
        show_wireframe: bool = True,
        show_com_rod: bool = True,
        show_alpha_point_rod: bool = True,
    ):
        """
        Show mesh with camera view perpendicular to the alpha plane.
        
        Args:
            mesh_path: Path to the mesh file
            center_of_mass: Center of mass of the mesh
            alpha_plane_normal: Normal vector to the alpha plane
            window_title: Title for the visualization window
            pedestal_points: Optional array of pedestal points to display
            alpha_angle_point: Optional point that produces the minimum alpha angle (to highlight)
            show_wireframe: Render mesh wireframe overlay
            show_com_rod: Draw the center-of-mass rod (pointing toward the camera)
            show_alpha_point_rod: Draw the alpha-point rod (pointing toward the camera)
        """
        try:
            import open3d as o3d
            import numpy as np
            
            width, height, left, top = PointCloudVisualization._get_visualizer_dimensions()

            # Load the mesh
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            if len(mesh.vertices) == 0:
                print(f"Warning: Could not load mesh from {mesh_path}")
                return

            mesh.compute_vertex_normals()

            # Use mesh bounds to scale visualization helpers (center marker, rod, etc.)
            bbox = mesh.get_axis_aligned_bounding_box()
            bbox_extent = np.asarray(bbox.get_extent())
            bbox_diag = float(np.linalg.norm(bbox_extent)) if bbox_extent.size else 0.0
            if bbox_diag == 0.0:
                bbox_diag = 1.0
            marker_radius = max(0.02, bbox_diag * 0.015)
            rod_half_length = max(0.1, bbox_diag * 0.45)

            # Create list of geometries to display
            geometries = [mesh]

            if show_wireframe and len(mesh.triangles) > 0:
                wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
                wireframe.paint_uniform_color([0.0, 0.0, 0.0])
                geometries.append(wireframe)
            
            # Add pedestal points if provided
            if pedestal_points is not None and len(pedestal_points) > 0:
                pedestal_pcd = o3d.geometry.PointCloud()
                pedestal_pcd.points = o3d.utility.Vector3dVector(pedestal_points)
                # Color pedestal points blue
                pedestal_pcd.paint_uniform_color([0.0, 0.0, 1.0])  # Blue
                # estimate normals for better visualization
                pedestal_pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50)
                )
                geometries.append(pedestal_pcd)
                print(f"Added {len(pedestal_points)} pedestal points (blue)")
            
            center_of_mass = np.asarray(center_of_mass, dtype=float)
            lookat = center_of_mass
            front = np.asarray(alpha_plane_normal, dtype=float)
            front_norm = float(np.linalg.norm(front))
            if not np.isfinite(front_norm) or front_norm == 0.0:
                front = np.array([0.0, 0.0, -1.0])
                front_norm = 1.0
            front = front / front_norm
            # Open3D expects `front` to describe the vector from the lookat point toward the camera.
            # This means a helper that should extend toward the viewer must follow `front`, not its negative.
            line_of_sight = front

            # Add a visible marker at the center of mass so it stays apparent even inside the rock
            com_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=marker_radius)
            com_sphere.translate(center_of_mass)
            com_sphere.paint_uniform_color([1.0, 1.0, 0.0])  # Yellow for visibility
            com_sphere.compute_vertex_normals()
            geometries.append(com_sphere)

            marker_tip_offset = max(marker_radius * 1.2, bbox_diag * 0.02)
            com_tip = center_of_mass + line_of_sight * marker_tip_offset
            com_tip_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=marker_radius * 0.75)
            com_tip_sphere.translate(com_tip)
            com_tip_sphere.paint_uniform_color([1.0, 1.0, 0.0])
            com_tip_sphere.compute_vertex_normals()
            geometries.append(com_tip_sphere)

            end_cap_radius = max(0.015, marker_radius * 0.6)
            rod_length = max(rod_half_length * 2.0, bbox_diag * 0.65)

            if show_com_rod:
                rod_points = np.vstack([
                    center_of_mass,
                    center_of_mass + line_of_sight * rod_length,
                ])
                rod = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(rod_points),
                    lines=o3d.utility.Vector2iVector([[0, 1]]),
                )
                rod.paint_uniform_color([1.0, 1.0, 0.0])  # Yellow rod for contrast
                geometries.append(rod)

                # Cap only the far end so the rod exits the rock visibly
                com_cap = o3d.geometry.TriangleMesh.create_sphere(radius=end_cap_radius)
                com_cap.translate(rod_points[1])
                com_cap.paint_uniform_color([1.0, 1.0, 0.0])
                com_cap.compute_vertex_normals()
                geometries.append(com_cap)

            # Add alpha angle point as a highlighted sphere (and rod) if provided
            if alpha_angle_point is not None:
                alpha_angle_point = np.asarray(alpha_angle_point, dtype=float)
                alpha_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=max(0.02, marker_radius * 0.75))
                alpha_sphere.translate(alpha_angle_point)
                alpha_sphere.paint_uniform_color([0.0, 1.0, 0.0])  # Bright green
                alpha_sphere.compute_vertex_normals()
                geometries.append(alpha_sphere)

                alpha_tip = alpha_angle_point + line_of_sight * marker_tip_offset
                alpha_tip_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=max(0.015, marker_radius * 0.5))
                alpha_tip_sphere.translate(alpha_tip)
                alpha_tip_sphere.paint_uniform_color([0.0, 1.0, 0.0])
                alpha_tip_sphere.compute_vertex_normals()
                geometries.append(alpha_tip_sphere)

                if show_alpha_point_rod:
                    alpha_rod_points = np.vstack([
                        alpha_angle_point,
                        alpha_angle_point + line_of_sight * rod_length,
                    ])
                    alpha_rod = o3d.geometry.LineSet(
                        points=o3d.utility.Vector3dVector(alpha_rod_points),
                        lines=o3d.utility.Vector2iVector([[0, 1]]),
                    )
                    alpha_rod.paint_uniform_color([0.0, 1.0, 0.0])
                    geometries.append(alpha_rod)

                    alpha_cap = o3d.geometry.TriangleMesh.create_sphere(radius=end_cap_radius)
                    alpha_cap.translate(alpha_rod_points[1])
                    alpha_cap.paint_uniform_color([0.0, 1.0, 0.0])
                    alpha_cap.compute_vertex_normals()
                    geometries.append(alpha_cap)
                    print(f"Added alpha angle point sphere and rod at {alpha_angle_point} (bright green)")
            
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
                "width": width,
                "height": height,
                "lookat": lookat,
                "up": up,
                "front": front,
                "zoom": zoom,
                "left": left,
                "top": top,
            }
            
            # Show all geometries with alpha view
            o3d.visualization.draw_geometries(geometries, **vis_params)
            
        except Exception as e:
            print(f"Error in alpha view visualization: {e}")
            import traceback
            traceback.print_exc()