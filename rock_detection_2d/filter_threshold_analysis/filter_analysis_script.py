import os
import laspy
import numpy as np
import open3d as o3d
from utils import filter_point_cloud
from sklearn.cluster import DBSCAN
from joblib import Parallel, delayed
from tqdm import tqdm

def analyze_filters(input_dir, output_dir, padding=0.2, height_range=(0.3, 5.0), min_volume=0.5, n_jobs=None):
    """Analyze filters and set intensity values based on rejection criteria."""
    if n_jobs is None:
        n_jobs = os.cpu_count() - 1  # Use all but one core
        if n_jobs < 1:
            n_jobs = 1
    print(f"Using {n_jobs} parallel jobs.")
    os.makedirs(output_dir, exist_ok=True)
    filter_labels = {
        'noise_filter': 0,
        'height_filter': 1,
        'height_width_ratio_filter': 2,
        'density_filter': 3,
        'cluster_filter': 4,
        'small_cluster_filter': 5,
        'volume_filter': 6,
        'shape_filter': 7,
        'ground_contact_filter': 8,
        'passed_all_filters': 9
    }

    # Include both .las and .laz files
    las_files = [f for f in os.listdir(input_dir) if f.endswith(('.las', '.laz'))]

    def process_file(file_name):
        file_path = os.path.join(input_dir, file_name)
        print(f"Processing: {file_name}")

        try:
            # Read point cloud
            las = laspy.read(file_path)
            points = np.vstack((las.x, las.y, las.z)).transpose()
            intensities = np.zeros(len(points), dtype=np.uint8)  # Default intensity is 0

            # Convert to Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            # 1. Noise removal (Vertical and SOR filters)
            try:
                pcd, _ = filter_point_cloud(pcd, filter_type='sor', use_vertical_filter=True, vertical_std=1.5)
                pcd, _ = filter_point_cloud(pcd, filter_type='sor', k_neighbors=50, std_ratio=2.0)
            except Exception as e:
                print(f"Error during noise removal: {str(e)}")
                intensities[:] = filter_labels['noise_filter']
                save_las_with_intensity(las, intensities, os.path.join(output_dir, file_name))
                return

            # 2. Height filter
            filtered_points = np.asarray(pcd.points)
            min_bounds = np.min(filtered_points, axis=0)
            max_bounds = np.max(filtered_points, axis=0)
            height = max_bounds[2] - min_bounds[2]
            if height < height_range[0] or height > height_range[1]:
                intensities[:] = filter_labels['height_filter']
                save_las_with_intensity(las, intensities, os.path.join(output_dir, file_name))
                return

            # 3. Height/Width ratio filter
            x_size = max_bounds[0] - min_bounds[0] - (2 * padding)
            y_size = max_bounds[1] - min_bounds[1] - (2 * padding)
            width = max(x_size, y_size)
            if width <= 0 or height / width > 2.0:
                intensities[:] = filter_labels['height_width_ratio_filter']
                save_las_with_intensity(las, intensities, os.path.join(output_dir, file_name))
                return

            # 4. Point density check
            density = compute_point_density(pcd)
            if density < 17:
                intensities[:] = filter_labels['density_filter']
                save_las_with_intensity(las, intensities, os.path.join(output_dir, file_name))
                return

            # 5. Cluster check
            labels = cluster_points(np.asarray(pcd.points))
            unique_labels, counts = np.unique(labels, return_counts=True)
            if len(unique_labels) > 5:
                intensities[:] = filter_labels['cluster_filter']
                save_las_with_intensity(las, intensities, os.path.join(output_dir, file_name))
                return

            # 6. Largest cluster size relative to total points
            largest_cluster_idx = np.argmax(counts)
            largest_cluster_size = counts[largest_cluster_idx]
            if largest_cluster_size / len(points) < 0.5:
                intensities[:] = filter_labels['small_cluster_filter']
                save_las_with_intensity(las, intensities, os.path.join(output_dir, file_name))
                return

            # 7. Volume filter
            volume = estimate_volume(pcd)
            if volume < min_volume:
                intensities[:] = filter_labels['volume_filter']
                save_las_with_intensity(las, intensities, os.path.join(output_dir, file_name))
                return

            # 8. Geometric shape analysis
            main_cluster_mask = labels == unique_labels[largest_cluster_idx]
            main_cluster_points = np.asarray(pcd.points)[main_cluster_mask]
            pcd_filtered = o3d.geometry.PointCloud()
            pcd_filtered.points = o3d.utility.Vector3dVector(main_cluster_points)
            if not is_rock_shaped(pcd_filtered):
                intensities[:] = filter_labels['shape_filter']
                save_las_with_intensity(las, intensities, os.path.join(output_dir, file_name))
                return

            # 9. Ground contact check
            if not has_ground_contact(pcd_filtered):
                intensities[:] = filter_labels['ground_contact_filter']
                save_las_with_intensity(las, intensities, os.path.join(output_dir, file_name))
                return

            # Passed all filters
            intensities[:] = filter_labels['passed_all_filters']
            save_las_with_intensity(las, intensities, os.path.join(output_dir, file_name))

        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")

    # Parallel processing with progress bar
    Parallel(n_jobs=n_jobs)(
        delayed(process_file)(file_name) for file_name in tqdm(las_files, desc="Processing LAS/LAZ files")
    )


def estimate_volume(pcd, alpha=0.5):
    """Estimate volume of point cloud using alpha shapes."""
    try:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        if mesh.is_watertight():
            return mesh.get_volume()
        return 0
    except:
        return 0


def compute_point_density(pcd, radius=0.1):
    """Compute local point density for each point."""
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    densities = []
    points = np.asarray(pcd.points)

    for i in range(len(points)):
        [k, _, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], radius)
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

    covariance = np.cov(points, rowvar=False)
    eigenvalues, _ = np.linalg.eigh(covariance)
    eigenvalues = np.sort(eigenvalues)[::-1]

    return eigenvalues[2] / eigenvalues[0] >= min_eigenvalue_ratio


def has_ground_contact(pcd, ground_threshold=0.05):
    """Check if point cloud has contact with the ground plane."""
    points = np.asarray(pcd.points)
    min_z = np.min(points[:, 2])
    ground_points = points[points[:, 2] < min_z + ground_threshold]
    return len(ground_points) >= 0.05 * len(points)


def save_las_with_intensity(las, intensities, output_path):
    """Save LAS file with updated intensity values."""
    las.intensity = intensities
    las.write(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze filters and set intensity values.")
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory containing LAS files.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory to save processed LAS files.")
    parser.add_argument("--padding", type=float, default=0.2, help="Padding value used in rock extraction.")
    parser.add_argument("--min-height", type=float, default=0.3, help="Minimum rock height in meters.")
    parser.add_argument("--max-height", type=float, default=5.0, help="Maximum rock height in meters.")

    args = parser.parse_args()

    analyze_filters(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        padding=args.padding,
        height_range=(args.min_height, args.max_height)
    )
