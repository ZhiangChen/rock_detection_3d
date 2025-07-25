import os
import laspy
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from datetime import datetime
import open3d as o3d
from utils import filter_point_cloud, compute_rough_dimensions
from sklearn.cluster import DBSCAN
from filter_stats import FilterStats
import copy  
from joblib import Parallel, delayed 
import logging 

# Suppress Open3D warnings
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

stats = FilterStats()

def process_rock(file_path, padding=0.2, height_range=(0.3, 5.0), debug_viz=False):
    """Process a single rock with filtering and dimension calculation."""
    try:
        # print(f"\nProcessing rock: {os.path.basename(file_path)}")
        
        # Read point cloud
        las = laspy.read(file_path)
        points = np.vstack((las.x, las.y, las.z)).transpose()
        # print(f"Loaded point cloud with {len(points)} points")
        
        # Convert to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Apply filters
        try:
            # 1. Vertical filter
            pcd_before = copy.deepcopy(pcd)  # Use deepcopy instead of clone
            pcd, _ = filter_point_cloud(pcd, filter_type='sor', use_vertical_filter=True, vertical_std=1.5)
            stats.log_filter('vertical_filter', len(pcd_before.points), len(pcd.points), pcd_before, pcd, debug_viz)

            # 2. Statistical Outlier Removal (SOR) filter
            pcd_before = copy.deepcopy(pcd)  # Use deepcopy instead of clone
            pcd, _ = filter_point_cloud(pcd, filter_type='sor', k_neighbors=100, std_ratio=1.0)
            stats.log_filter('sor_filter', len(pcd_before.points), len(pcd.points), pcd_before, pcd, debug_viz)

            # 3. Height filter
            filtered_points = np.asarray(pcd.points)
            min_bounds = np.min(filtered_points, axis=0)
            max_bounds = np.max(filtered_points, axis=0)
            height = max_bounds[2] - min_bounds[2]
            if height < height_range[0] or height > height_range[1]:
                stats.log_rejection('height_out_of_range', os.path.basename(file_path))
                return None

            # 4. Height/Width ratio filter
            x_size = max_bounds[0] - min_bounds[0] - (2 * padding)
            y_size = max_bounds[1] - min_bounds[1] - (2 * padding)
            width = max(x_size, y_size)
            if width <= 0 or height / width > 1.5:
                stats.log_rejection('extreme_height_width_ratio', os.path.basename(file_path))
                return None

            # 5. Volume filter
            volume = estimate_volume(pcd)
            if volume > 10:
                stats.log_rejection('volume_too_large', os.path.basename(file_path))
                return None

            # 6. Density filter
            density = compute_point_density(pcd)
            if density <= 19:
                stats.log_rejection('low_density', os.path.basename(file_path))
                return None

            # 7. Cluster check
            labels = cluster_points(np.asarray(pcd.points))
            unique_labels, counts = np.unique(labels, return_counts=True)
            if len(unique_labels) >= 5:
                stats.log_rejection('too_many_clusters', os.path.basename(file_path))
                return None

            # # 8. Eigenvalue ratio filter
            # covariance = np.cov(filtered_points, rowvar=False)
            # eigenvalues = np.sort(np.linalg.eigvals(covariance))[::-1]
            # if eigenvalues[2] / eigenvalues[0] <= 0.08:
            #     stats.log_rejection('low_eigenvalue_ratio', os.path.basename(file_path))
            #     return None

            # 9. Normal consistency filter
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            normals = np.asarray(pcd.normals)
            consistencies = []
            sample_size = min(1000, len(filtered_points))
            indices = np.random.choice(len(filtered_points), sample_size, replace=False)
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    dot_product = np.abs(np.dot(normals[indices[i]], normals[indices[j]]))
                    consistencies.append(dot_product)
            avg_consistency = np.mean(consistencies)
            if avg_consistency >= 0.7:
                stats.log_rejection('high_normal_consistency', os.path.basename(file_path))
                return None

        except Exception as e:
            print(f"Error during filtering: {str(e)}")
            return None
            
        return {
            'pbr_name': os.path.splitext(os.path.basename(file_path))[0],
            'pbr_location': file_path,
            'height': height,
            'width': width,
            'length': min(x_size, y_size),
            'height_width_ratio': height / width,
        }
        
    except Exception as e:
        print(f"Error processing rock: {str(e)}")
        return None


def compute_point_density(pcd, radius=0.1):
    """Compute local point density for each point."""
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    densities = []
    points = np.asarray(pcd.points)
    
    for i in range(len(points)):
        [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], radius)
        densities.append(k)
            
    return np.mean(densities)


def cluster_points(points, eps=0.1, min_samples=10):
    """Cluster points using DBSCAN to identify connected components."""
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    return clustering.labels_


def is_rock_shaped(pcd, min_eigenvalue_ratio=0.1):
    """Analyze if point cloud has rock-like geometric properties using PCA."""
    points = np.asarray(pcd.points)
    if len(points) < 10:
        return False
        
    # Compute covariance matrix and its eigenvalues
    covariance = np.cov(points, rowvar=False)
    eigenvalues, _ = np.linalg.eigh(covariance)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Sort in descending order
    
    # Check if the shape is not too flat or linear
    if eigenvalues[2] / eigenvalues[0] < min_eigenvalue_ratio:
        return False
        
    # Compute surface normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    normals = np.asarray(pcd.normals)
    
    # Measure normal consistency (dot product between normals)
    consistency = 0
    sample_size = min(1000, len(points))
    indices = np.random.choice(len(points), sample_size, replace=False)
    
    for i in range(len(indices)):
        for j in range(i+1, len(indices)):
            dot_product = np.abs(np.dot(normals[indices[i]], normals[indices[j]]))
            consistency += dot_product
            
    avg_consistency = consistency / (sample_size * (sample_size - 1) / 2)
    return avg_consistency < 0.7  # Threshold for normal consistency


def estimate_volume(pcd, alpha=0.5):
    """Estimate volume of point cloud using alpha shapes."""
    try:
        # Try to compute volume using alpha shapes
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        if mesh.is_watertight():
            return mesh.get_volume()
        
        # Fallback to convex hull if alpha shape fails
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_convex_hull(pcd)
        return mesh.get_volume()
    except:
        # Fallback to simple bounding box volume estimation
        points = np.asarray(pcd.points)
        min_bounds = np.min(points, axis=0)
        max_bounds = np.max(points, axis=0)
        return np.prod(max_bounds - min_bounds)


def generate_pbr_database(input_dir, output_file, max_rocks=100, height_range=(0.3, 5.0), 
                         padding=0.2, debug_viz=False):
    """Generate PBR database after filtering and selecting most fragile rocks."""
    print(f"Analyzing rocks in: {input_dir}")
    
    # Ensure output_file is a valid file path
    if os.path.isdir(output_file):
        output_file = os.path.join(output_file, "pbr_database.csv")
        print(f"Output file path adjusted to: {output_file}")
    
    # Get all LAZ files
    las_files = [f for f in os.listdir(input_dir) if f.endswith('.laz')]
    if not las_files:
        print("No .laz files found!")
        return
    
    # Process rocks with filtering in parallel
    print("Processing rocks with filtering and calculating ratios...")
    num_cores = os.cpu_count()-1  # Use appropriate number of cores
    rock_data = Parallel(n_jobs=num_cores)(
        delayed(process_rock)(os.path.join(input_dir, las_file), padding, height_range, debug_viz)
        for las_file in tqdm(las_files, desc="Processing rocks")
    )
    
    # Filter out None results
    rock_data = [result for result in rock_data if result is not None]
    
    if not rock_data:
        print("No valid rocks found after filtering!")
        return
        
    # Convert to DataFrame and sort by fragility (H/W ratio)
    df = pd.DataFrame(rock_data)
    df = df.sort_values('height_width_ratio', ascending=False)
    
    # Select top N most fragile rocks
    df = df.head(max_rocks)
    
    # Add empty columns for future use
    empty_columns = [
        'segmented_pbr_location', 'mesh_reconstruction_location',
        'false_positive', 'processed', 'center_of_mass',
        'major_orientations', 'height_width_face', 'length_width_ratio',
        'length_width_face', 'alpha_angle', 'alpha_rectangular',
        'beta_angle', 'smoothness_threshold', 'curvature_threshold',
        'proximity_threshold'
    ]
    
    for col in empty_columns:
        df[col] = ''
    
    # Save database
    df.to_csv(output_file, index=False)
    print(f"\nDatabase generated successfully!")
    print(f"Total rocks analyzed: {len(las_files)}")
    print(f"Valid rocks after height filtering: {len(rock_data)}")
    print(f"Selected PBRs (most fragile): {len(df)}")
    print(f"Height range: {height_range[0]}-{height_range[1]} meters")
    print(f"Database saved to: {output_file}")

    # After processing all rocks, generate analysis report
    stats.generate_report()
    print("\nFilter analysis report generated in 'filter_analysis' directory")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate PBR database from point cloud files')
    parser.add_argument('--input', type=str, required=True,
                        help='Directory containing rock point clouds')
    parser.add_argument('--output', type=str, default='pbr_database.csv',
                        help='Output CSV file path')
    parser.add_argument('--max-rocks', type=int, default=100,
                        help='Number of most fragile rocks to keep')
    parser.add_argument('--min-height', type=float, default=0.3,
                        help='Minimum rock height in meters')
    parser.add_argument('--max-height', type=float, default=5.0,
                        help='Maximum rock height in meters')
    parser.add_argument('--padding', type=float, default=0.2,
                        help='Padding value used in rock extraction')
    parser.add_argument('--debug-viz', action='store_true',
                       help='Enable visualization of filter steps')
    
    args = parser.parse_args()
    
    generate_pbr_database(
        args.input,
        args.output,
        max_rocks=args.max_rocks,
        height_range=(args.min_height, args.max_height),
        padding=args.padding,
        debug_viz=args.debug_viz
    )
