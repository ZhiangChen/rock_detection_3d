import numpy as np
import open3d as o3d
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import logging

def process_point_chunk_noise(points, normals, pcd_tree, start_idx, end_idx, k_neighbors, max_error, relative_error):
    """Helper function to process a chunk of points for noise filter"""
    chunk_inliers = []
    
    for i in range(start_idx, end_idx):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(points[i], k_neighbors)
        if k < 3:  # Need at least 3 points to fit a plane
            continue
        
        neighbors = points[idx]
        centroid = np.mean(neighbors, axis=0)
        centered = neighbors - centroid
        
        cov = np.dot(centered.T, centered) / len(centered)
        eigenvals, eigenvects = np.linalg.eigh(cov)
        normal = eigenvects[:, 0]
        
        distances = np.abs(np.dot(centered, normal))
        local_variation = np.std(distances) if relative_error else 1.0
        threshold = max_error * local_variation
        
        point_distance = np.abs(np.dot(points[i] - centroid, normal))
        if point_distance <= threshold:
            chunk_inliers.append(i)
            
    return chunk_inliers

def noise_filter(pcd, k_neighbors=30, max_error=0.1, relative_error=True, n_threads=8):
    """CloudCompare-style noise filter using KNN search and parallel processing."""
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals) if pcd.has_normals() else None
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    
    n_points = len(points)
    chunk_size = max(1000, n_points // (n_threads * 10))
    chunks = [(i, min(i + chunk_size, n_points)) for i in range(0, n_points, chunk_size)]
    
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        process_func = partial(
            process_point_chunk_noise,
            points, normals, pcd_tree,
            k_neighbors=k_neighbors,
            max_error=max_error,
            relative_error=relative_error
        )
        chunk_results = executor.map(lambda x: process_func(*x), chunks)
    
    inlier_indices = np.array([i for chunk in chunk_results for i in chunk])
    filtered_pcd = pcd.select_by_index(inlier_indices)
    return filtered_pcd, inlier_indices

def sor_filter(pcd, k_neighbors=6, std_ratio=2.0, n_threads=8):
    """Statistical Outlier Removal filter using KNN search and parallel processing."""
    points = np.asarray(pcd.points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    
    def process_chunk(start_idx, end_idx):
        distances = []
        indices = []
        for i in range(start_idx, end_idx):
            [k, idx, dist] = pcd_tree.search_knn_vector_3d(points[i], k_neighbors + 1)
            avg_dist = np.mean(np.sqrt(dist[1:]))  # Exclude self-distance
            distances.append(avg_dist)
            indices.append(i)
        return indices, distances
    
    n_points = len(points)
    chunk_size = max(1000, n_points // (n_threads * 10))
    chunks = [(i, min(i + chunk_size, n_points)) for i in range(0, n_points, chunk_size)]
    
    all_distances = []
    all_indices = []
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(process_chunk, start, end) for start, end in chunks]
        for future in futures:
            indices, distances = future.result()
            all_indices.extend(indices)
            all_distances.extend(distances)
    
    distances = np.array(all_distances)
    indices = np.array(all_indices)
    
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    threshold = mean_dist + std_ratio * std_dist
    
    mask = distances < threshold
    inlier_indices = indices[mask]
    filtered_pcd = pcd.select_by_index(inlier_indices)
    
    return filtered_pcd, inlier_indices

def vertical_outlier_filter(pcd, std_multiplier=1.0):
    """Filter points that are too high above the main rock body.
    
    Args:
        pcd: Open3D point cloud
        std_multiplier: Number of standard deviations to use as threshold
        
    Returns:
        tuple: (filtered point cloud, inlier indices)
    """
    points = np.asarray(pcd.points)
    z_values = points[:, 2]
    
    mean_z = np.mean(z_values)
    std_z = np.std(z_values)
    
    # Keep points within std_multiplier standard deviations above mean
    threshold = mean_z + (std_multiplier * std_z)
    inlier_mask = z_values <= threshold
    inlier_indices = np.where(inlier_mask)[0]
    
    filtered_pcd = pcd.select_by_index(inlier_indices)
    return filtered_pcd, inlier_indices

def compute_rough_dimensions(pcd, padding=0.2):
    """Compute approximate height/width ratio from point cloud bounds with padding.
    
    Args:
        pcd: Open3D point cloud
        padding: Amount of padding relative to dimensions (similar to rock extraction)
        
    Returns:
        tuple: (height/width ratio, height, width, padded_dimensions)
    """
    points = np.asarray(pcd.points)
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    
    # Get raw dimensions
    dimensions = max_bound - min_bound
    
    # Apply padding (similar to rock_extraction_2d_pipeline.py)
    padded_min = min_bound - (dimensions * padding)
    padded_max = max_bound + (dimensions * padding)
    padded_dimensions = padded_max - padded_min
    
    # Get height and width from padded dimensions
    height = padded_dimensions[2]  # Z-axis height
    width = min(padded_dimensions[0], padded_dimensions[1])  # Smaller of X and Y dimensions
    
    hw_ratio = height / width if width > 0 else float('inf')
    return hw_ratio, height, width, padded_dimensions

def filter_point_cloud(pcd, filter_type='sor', use_vertical_filter=False, 
                      k_neighbors=6, std_ratio=2.0, max_error=0.1, 
                      relative_error=True, vertical_std=1.0, n_threads=8,
                      height_width_threshold=2.0):
    """Combined filtering function that can apply multiple filters.
    
    Args:
        pcd: Open3D point cloud
        filter_type: Type of filter ('sor' or 'noise')
        use_vertical_filter: Whether to apply vertical outlier filtering
        vertical_std: Number of standard deviations for vertical filter
        height_width_threshold: Minimum H/W ratio to apply vertical filter
        ...other filter parameters...
    
    Returns:
        tuple: (filtered point cloud, inlier indices)
    """
    current_pcd = pcd
    all_inliers = np.arange(len(np.asarray(pcd.points)))
    
    # Apply vertical filter only if the rock is tall enough
    if use_vertical_filter:
        hw_ratio, height, width, _ = compute_rough_dimensions(current_pcd)
        should_apply_vertical = hw_ratio > height_width_threshold
        
        if should_apply_vertical:
            logging.info(f"Rock is tall (H/W ratio = {hw_ratio:.2f}), applying vertical filter")
            current_pcd, v_inliers = vertical_outlier_filter(current_pcd, vertical_std)
            all_inliers = all_inliers[v_inliers]
        else:
            logging.info(f"Rock is not tall enough (H/W ratio = {hw_ratio:.2f}), skipping vertical filter")
    
    # Apply main filter
    if filter_type == 'sor':
        current_pcd, f_inliers = sor_filter(
            current_pcd, k_neighbors=k_neighbors, std_ratio=std_ratio, n_threads=n_threads
        )
    else:
        current_pcd, f_inliers = noise_filter(
            current_pcd, k_neighbors=k_neighbors, max_error=max_error, 
            relative_error=relative_error, n_threads=n_threads
        )
    
    # Update indices to reference original point cloud
    final_inliers = all_inliers[f_inliers]
    return current_pcd, final_inliers
