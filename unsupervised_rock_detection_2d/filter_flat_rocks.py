import os
import numpy as np
import laspy
import shutil
import argparse
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def log_message(message):
    """Print message with timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

def calculate_flatness(file_path, flatness_threshold=0.05, min_points=20):
    """
    Determine if a point cloud is flat by comparing Z-range to XY-extent.
    
    Args:
        file_path: Path to the LAS/LAZ file
        flatness_threshold: Threshold below which a rock is considered flat
                           (Z-range relative to XY diagonal)
        min_points: Minimum number of points for valid analysis
        
    Returns:
        is_flat: Boolean indicating if the rock is flat
        flatness_ratio: The calculated Z-range to XY-diagonal ratio
        metrics: Dictionary with point cloud metrics
    """
    try:
        # Load point cloud
        point_cloud = laspy.read(file_path)
        
        # Check if there are enough points
        if len(point_cloud) < min_points:
            return True, 0.0, {"points": len(point_cloud), "error": "Too few points"}
        
        # Get coordinates
        x, y, z = point_cloud.x, point_cloud.y, point_cloud.z
        
        # Calculate ranges
        x_range = np.max(x) - np.min(x)
        y_range = np.max(y) - np.min(y)
        z_range = np.max(z) - np.min(z)
        
        # Calculate XY diagonal (approximation of horizontal extent)
        xy_diagonal = np.sqrt(x_range**2 + y_range**2)
        
        # Avoid division by zero
        if xy_diagonal == 0:
            return True, 0.0, {"points": len(point_cloud), "error": "Zero horizontal extent"}
        
        # Calculate flatness ratio: Z-range relative to XY extent
        flatness_ratio = z_range / xy_diagonal
        
        # Calculate improved metrics for flatness detection
        # 1. Standard deviation of Z relative to XY extent
        std_z = np.std(z)
        std_z_ratio = std_z / xy_diagonal
        
        # 2. IQR of Z (less sensitive to outliers)
        q75, q25 = np.percentile(z, [75, 25])
        z_iqr = q75 - q25
        z_iqr_ratio = z_iqr / xy_diagonal
        
        # 3. Planar fit error (PCA-based)
        # Center the points
        centered_points = np.column_stack([x - np.mean(x), y - np.mean(y), z - np.mean(z)])
        # Calculate covariance matrix
        cov_matrix = np.cov(centered_points.T)
        # Eigenvalues of covariance matrix
        eigenvalues = np.linalg.eigvals(cov_matrix)
        # Sort eigenvalues (smallest first)
        eigenvalues = np.sort(eigenvalues)
        # Planarity measure: ratio of smallest eigenvalue to sum
        planarity = eigenvalues[0] / np.sum(eigenvalues) if np.sum(eigenvalues) > 0 else 0
        
        metrics = {
            "points": len(point_cloud),
            "x_range": x_range,
            "y_range": y_range,
            "z_range": z_range,
            "xy_diagonal": xy_diagonal,
            "flatness_ratio": flatness_ratio,
            "std_z": std_z,
            "std_z_ratio": std_z_ratio,
            "z_iqr": z_iqr,
            "z_iqr_ratio": z_iqr_ratio,
            "planarity": planarity
        }
        
        # Use ONLY the flatness_ratio for consistent classification
        # A rock is flat if its height-to-width ratio is below the threshold
        is_flat = flatness_ratio < flatness_threshold
        
        return is_flat, flatness_ratio, metrics
    
    except Exception as e:
        return False, 0.0, {"points": 0, "error": str(e)}

def analyze_specific_rocks(input_dir, rock_names, flatness_threshold=0.05, visualize=False):
    """
    Analyze specific rock point clouds to diagnose flatness detection issues.
    
    Args:
        input_dir: Directory containing rock point clouds
        rock_names: List of rock filenames to analyze
        flatness_threshold: Threshold for determining flatness
        visualize: Whether to visualize the point clouds
    """
    log_message(f"Analyzing specific rocks with threshold {flatness_threshold}:")
    
    for rock_name in rock_names:
        file_path = os.path.join(input_dir, rock_name)
        if not os.path.exists(file_path):
            log_message(f"❌ File not found: {file_path}")
            continue
            
        try:
            # Load point cloud
            point_cloud = laspy.read(file_path)
            x, y, z = point_cloud.x, point_cloud.y, point_cloud.z
            
            # Calculate metrics
            is_flat, flatness_ratio, metrics = calculate_flatness(file_path, flatness_threshold)
            
            # Print detailed metrics
            log_message(f"\n📊 Analysis for {rock_name}:")
            log_message(f"  Points: {metrics['points']}")
            log_message(f"  X range: {metrics['x_range']:.4f}")
            log_message(f"  Y range: {metrics['y_range']:.4f}")
            log_message(f"  Z range: {metrics['z_range']:.4f}")
            log_message(f"  XY diagonal: {metrics['xy_diagonal']:.4f}")
            log_message(f"  Flatness ratio (Z/XY): {metrics['flatness_ratio']:.4f}" + 
                       f" {'FLAT' if metrics['flatness_ratio'] < flatness_threshold else 'NOT FLAT'}")
            log_message(f"  Z std dev: {metrics['std_z']:.4f}")
            log_message(f"  Z std dev / XY: {metrics['std_z_ratio']:.4f}")
            log_message(f"  Z IQR: {metrics['z_iqr']:.4f}")
            log_message(f"  Z IQR / XY: {metrics['z_iqr_ratio']:.4f}")
            log_message(f"  Planarity: {metrics['planarity']:.6f}")
            log_message(f"  Final classification: {'FLAT' if is_flat else 'NOT FLAT'}")
            
            # Visualize if requested
            if visualize:
                fig = plt.figure(figsize=(12, 10))
                
                # 3D scatter plot
                ax1 = fig.add_subplot(2, 2, 1, projection='3d')
                ax1.scatter(x, y, z, c=z, cmap='viridis', s=1, alpha=0.5)
                ax1.set_title(f"{rock_name} - 3D View")
                ax1.set_xlabel('X')
                ax1.set_ylabel('Y')
                ax1.set_zlabel('Z')
                
                # Top view (X-Y)
                ax2 = fig.add_subplot(2, 2, 2)
                ax2.scatter(x, y, c=z, cmap='viridis', s=1, alpha=0.5)
                ax2.set_title('Top View (X-Y)')
                ax2.set_xlabel('X')
                ax2.set_ylabel('Y')
                ax2.axis('equal')
                
                # Side view (X-Z)
                ax3 = fig.add_subplot(2, 2, 3)
                ax3.scatter(x, z, c=z, cmap='viridis', s=1, alpha=0.5)
                ax3.set_title('Side View (X-Z)')
                ax3.set_xlabel('X')
                ax3.set_ylabel('Z')
                
                # Side view (Y-Z)
                ax4 = fig.add_subplot(2, 2, 4)
                ax4.scatter(y, z, c=z, cmap='viridis', s=1, alpha=0.5)
                ax4.set_title('Side View (Y-Z)')
                ax4.set_xlabel('Y')
                ax4.set_ylabel('Z')
                
                plt.tight_layout()
                plt.savefig(f"{rock_name}_analysis.png")
                plt.close()
                log_message(f"Saved visualization to {rock_name}_analysis.png")
                
        except Exception as e:
            log_message(f"❌ Error analyzing {rock_name}: {str(e)}")

def filter_flat_rocks(input_dir, output_dir, flatness_threshold=0.05, min_points=20, dry_run=False):
    """
    Filter flat rocks from input directory and move them to output directory.
    
    Args:
        input_dir: Directory containing rock point clouds
        output_dir: Directory to move flat rocks to
        flatness_threshold: Threshold for determining flatness
        min_points: Minimum number of points for valid analysis
        dry_run: If True, don't actually move files, just report
    """
    start_time = datetime.now()
    log_message(f"Starting flat rock filtering process...")
    log_message(f"Source directory: {input_dir}")
    log_message(f"Destination directory: {output_dir}")
    log_message(f"Flatness threshold: {flatness_threshold}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir) and not dry_run:
        os.makedirs(output_dir)
        log_message(f"Created output directory: {output_dir}")
    
    # Get all LAS/LAZ files
    las_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.las', '.laz'))]
    log_message(f"Found {len(las_files)} point cloud files to analyze")
    
    # Statistics
    total_files = len(las_files)
    flat_files = 0
    error_files = 0
    
    # Process each file
    results = []
    
    for filename in tqdm(las_files, desc="Processing files"):
        file_path = os.path.join(input_dir, filename)
        
        # Check if file is flat
        is_flat, flatness_ratio, metrics = calculate_flatness(
            file_path, 
            flatness_threshold=flatness_threshold,
            min_points=min_points
        )
        
        # Record result
        result = {
            "filename": filename,
            "is_flat": is_flat,
            "flatness_ratio": flatness_ratio,
            **metrics
        }
        results.append(result)
        
        if "error" in metrics:
            error_files += 1
            log_message(f"Error processing {filename}: {metrics['error']}")
            continue
        
        # Move file if it's flat
        if is_flat:
            flat_files += 1
            if not dry_run:
                dest_path = os.path.join(output_dir, filename)
                try:
                    shutil.move(file_path, dest_path)
                except Exception as e:
                    log_message(f"Error moving {filename}: {str(e)}")
    
    # Sort results by flatness ratio
    results.sort(key=lambda x: x.get("flatness_ratio", 0))
    
    # Print summary
    log_message("\nSummary:")
    log_message(f"Total files analyzed: {total_files}")
    log_message(f"Flat rocks identified: {flat_files} ({flat_files/total_files*100:.1f}%)")
    log_message(f"Files with errors: {error_files}")
    
    if dry_run:
        log_message("DRY RUN - No files were actually moved")
    else:
        log_message(f"Flat rocks moved to: {output_dir}")
    
    # Show top 5 flattest rocks
    log_message("\nTop 5 flattest rocks:")
    for i, result in enumerate(results[:5]):
        log_message(f"{i+1}. {result['filename']}: {result['flatness_ratio']:.4f} " + 
                   f"(z_range: {result.get('z_range', 'N/A'):.3f}, " +
                   f"xy_diagonal: {result.get('xy_diagonal', 'N/A'):.3f})")
    
    # Show top 5 least flat rocks
    log_message("\nTop 5 least flat rocks:")
    for i, result in enumerate(results[-5:]):
        log_message(f"{i+1}. {result['filename']}: {result['flatness_ratio']:.4f} " +
                   f"(z_range: {result.get('z_range', 'N/A'):.3f}, " + 
                   f"xy_diagonal: {result.get('xy_diagonal', 'N/A'):.3f})")
    
    execution_time = (datetime.now() - start_time).total_seconds()
    log_message(f"\nExecution time: {execution_time:.2f} seconds")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter flat rocks from point cloud files')
    parser.add_argument('--input', type=str, default='',
                        help='Directory containing rock point clouds')
    parser.add_argument('--output', type=str, default='flat_rocks',
                        help='Directory to move flat rocks to')
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='Flatness threshold (Z-range/XY-diagonal ratio)')
    parser.add_argument('--min-points', type=int, default=20,
                        help='Minimum points needed in a cloud')
    parser.add_argument('--dry-run', action='store_true',
                        help='Analyze but do not move files')
    parser.add_argument('--analyze', nargs='*', default=None,
                        help='Analyze rock files (provide filenames or leave empty to analyze all)')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize point cloud when analyzing specific rocks')
    
    args = parser.parse_args()
    
    # Analyze specific rocks if requested
    if args.analyze is not None:
        input_dir = args.input or '.'  # Default to current directory if not specified
        
        # If no specific files given, get all LAS/LAZ files in the directory
        if len(args.analyze) == 0:
            rock_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.las', '.laz'))]
            log_message(f"Analyzing all {len(rock_files)} LAS/LAZ files in directory")
        else:
            rock_files = args.analyze
        
        analyze_specific_rocks(
            input_dir,
            rock_files,
            flatness_threshold=args.threshold,
            visualize=args.visualize
        )
    # Otherwise run the full filtering process
    elif args.input and args.output:
        filter_flat_rocks(
            args.input,
            args.output,
            flatness_threshold=args.threshold,
            min_points=args.min_points,
            dry_run=args.dry_run
        )
    else:
        parser.print_help()
