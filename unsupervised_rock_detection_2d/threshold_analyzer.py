import os
import numpy as np
import laspy
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict
from tqdm import tqdm  # Add tqdm import

class ThresholdAnalyzer:
    def __init__(self, laz_dir, positive_ids, negative_ids, output_dir="threshold_analysis"):
        self.laz_dir = laz_dir
        self.positive_ids = set(positive_ids)
        self.negative_ids = set(negative_ids)
        self.output_dir = output_dir
        self.metrics = defaultdict(lambda: {'positive': [], 'negative': []})
        self.all_metrics = []  # Store all metric values for each rock
        os.makedirs(output_dir, exist_ok=True)

    def analyze_point_cloud(self, file_path, label):
        """Analyze a single point cloud and collect all relevant metrics"""
        try:
            las = laspy.read(file_path)
            points = np.vstack((las.x, las.y, las.z)).transpose()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # Collect metrics
            metrics = {
                'rock_id': os.path.splitext(os.path.basename(file_path))[0],
                'label': label,
                'Density': self._compute_density(pcd),
                'Eigenvalue Ratio': self._compute_eigenvalue_ratio(pcd),
                'Normal Consistency': self._compute_normal_consistency(pcd),
                'Ground Contact Ratio': self._compute_ground_contact_ratio(pcd),
                'Volume': self._compute_volume(pcd),
                'Height': self._compute_height(points),
                'Height/Width Ratio': self._compute_height_width_ratio(points),
                'Cluster Count': self._compute_cluster_count(points)
            }
            
            # # Print metrics for the current rock
            # print(f"Analyzed {label} rock: {os.path.basename(file_path)}")
            # for metric_name, value in metrics.items():
            #     if metric_name not in ['rock_id', 'label']:
            #         print(f"  {metric_name}: {value:.4f}")
            
            # Store metrics
            self.all_metrics.append(metrics)
            return True
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return False

    def analyze_all(self):
        """Analyze all point clouds in the laz directory based on positive and negative IDs"""
        files = [file for file in os.listdir(self.laz_dir) if file.endswith('.laz') and file.startswith('rock_')]
        for file in tqdm(files, desc="Analyzing point clouds"):  # Add tqdm progress bar
            try:
                rock_id = int(file.split('_')[1].split('.')[0])
                if rock_id in self.positive_ids:
                    self.analyze_point_cloud(os.path.join(self.laz_dir, file), 'positive')
                elif rock_id in self.negative_ids:
                    self.analyze_point_cloud(os.path.join(self.laz_dir, file), 'negative')
            except ValueError:
                print(f"Skipping file with invalid format: {file}")

    def generate_report(self):
        """Generate analysis report with all values, summary statistics, and histograms"""
        # Save all metric values to a CSV
        all_metrics_df = pd.DataFrame(self.all_metrics)
        all_metrics_csv_path = os.path.join(self.output_dir, "all_metrics.csv")
        all_metrics_df.to_csv(all_metrics_csv_path, index=False)
        print(f"Saved all metrics to {all_metrics_csv_path}")

        # Calculate summary statistics
        summary_stats = {}
        for metric_name in all_metrics_df.columns:
            if metric_name in ['rock_id', 'label']:
                continue
            pos_values = all_metrics_df[all_metrics_df['label'] == 'positive'][metric_name]
            neg_values = all_metrics_df[all_metrics_df['label'] == 'negative'][metric_name]
            summary_stats[metric_name] = {
                'positive_mean': pos_values.mean(),
                'positive_std': pos_values.std(),
                'positive_min': pos_values.min(),
                'positive_max': pos_values.max(),
                'negative_mean': neg_values.mean(),
                'negative_std': neg_values.std(),
                'negative_min': neg_values.min(),
                'negative_max': neg_values.max(),
                'ks_statistic': stats.ks_2samp(pos_values, neg_values).statistic,
                'p_value': stats.ks_2samp(pos_values, neg_values).pvalue
            }

            # Remove extreme outliers for better visualization (max 5 outliers per group)
            def remove_extreme_outliers(values, max_outliers=5):
                if len(values) <= max_outliers:
                    return values
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR  # Use 3*IQR instead of 1.5*IQR for extreme outliers only
                upper_bound = Q3 + 3 * IQR
                
                # Identify outliers
                outliers_mask = (values < lower_bound) | (values > upper_bound)
                outliers = values[outliers_mask]
                
                # Remove only the most extreme outliers (up to max_outliers)
                if len(outliers) > max_outliers:
                    # Keep the max_outliers most extreme values as outliers to remove
                    lower_outliers = outliers[outliers < lower_bound]
                    upper_outliers = outliers[outliers > upper_bound]
                    
                    # Sort to get most extreme
                    lower_extreme = lower_outliers.nsmallest(min(len(lower_outliers), max_outliers//2))
                    upper_extreme = upper_outliers.nlargest(min(len(upper_outliers), max_outliers - len(lower_extreme)))
                    
                    extreme_outliers = pd.concat([lower_extreme, upper_extreme])
                    return values[~values.isin(extreme_outliers)]
                else:
                    return values[~outliers_mask]
            
            # Apply outlier removal
            pos_values_clean = remove_extreme_outliers(pos_values)
            neg_values_clean = remove_extreme_outliers(neg_values)

            # Compute common bin edges from both datasets
            all_values = np.concatenate([pos_values_clean, neg_values_clean])
            bins = np.histogram_bin_edges(all_values, bins=40)
            
            # Save histogram for the metric
            plt.figure(figsize=(6, 6))
            plt.hist(neg_values_clean, bins=bins, label='Negative', color='red', alpha=0.5)
            plt.hist(pos_values_clean, bins=bins, label='Positive', color='green', alpha=0.7)
            
            plt.title(f'{metric_name} Histogram (Extreme Outliers Removed)')
            plt.xlabel(metric_name)
            plt.ylabel('Frequency')
            
            # Calculate reasonable x-axis range
            min_val = min(pos_values_clean.min(), neg_values_clean.min())
            max_val = max(pos_values_clean.max(), neg_values_clean.max())
            plt.xticks(np.linspace(min_val, max_val, 15), rotation=45)  # Reduced tick density
            
            plt.legend()
            plt.tight_layout()  # Better layout adjustment
            save_name = metric_name.replace(' ', "_").replace('/', "_").lower() + '_histogram.png'
            histogram_path = os.path.join(self.output_dir, save_name)
            plt.savefig(histogram_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Print outlier removal info
            removed_pos = len(pos_values) - len(pos_values_clean)
            removed_neg = len(neg_values) - len(neg_values_clean)
            if removed_pos > 0 or removed_neg > 0:
                print(f"Removed {removed_pos} positive and {removed_neg} negative outliers from {metric_name} histogram")
            print(f"Saved histogram for {metric_name} to {histogram_path}")

        # Save summary statistics to a CSV
        summary_stats_df = pd.DataFrame.from_dict(summary_stats, orient='index')
        summary_stats_csv_path = os.path.join(self.output_dir, "summary_stats.csv")
        summary_stats_df.to_csv(summary_stats_csv_path)
        print(f"Saved summary statistics to {summary_stats_csv_path}")

    def _find_optimal_threshold(self, positive_values, negative_values):
        """Find optimal threshold that best separates positive and negative samples"""
        all_values = np.concatenate([positive_values, negative_values])
        best_threshold = None
        best_separation = -1
        
        for threshold in np.percentile(all_values, np.linspace(0, 100, 100)):
            tp = np.sum(np.array(positive_values) > threshold)
            tn = np.sum(np.array(negative_values) < threshold)
            separation = (tp/len(positive_values) + tn/len(negative_values)) / 2
            
            if separation > best_separation:
                best_separation = separation
                best_threshold = threshold
                
        return best_threshold

    # Helper methods for computing various metrics
    def _compute_density(self, pcd, radius=0.1):
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        points = np.asarray(pcd.points)
        densities = []
        for i in range(len(points)):
            [k, _, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], radius)
            densities.append(k)
        return np.mean(densities)

    def _compute_eigenvalue_ratio(self, pcd):
        covariance = np.cov(np.asarray(pcd.points), rowvar=False)
        eigenvalues = np.sort(np.linalg.eigvals(covariance))[::-1]
        return eigenvalues[2] / eigenvalues[0]

    def _compute_normal_consistency(self, pcd):
        pcd.estimate_normals()
        normals = np.asarray(pcd.normals)
        sample_size = min(1000, len(pcd.points))
        indices = np.random.choice(len(pcd.points), sample_size, replace=False)
        consistencies = []
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                dot_product = np.abs(np.dot(normals[indices[i]], normals[indices[j]]))
                consistencies.append(dot_product)
        return np.mean(consistencies)

    def _compute_ground_contact_ratio(self, pcd, threshold=0.05):
        points = np.asarray(pcd.points)
        min_z = np.min(points[:, 2])
        ground_points = points[points[:, 2] < min_z + threshold]
        return len(ground_points) / len(points)

    def _compute_volume(self, pcd):
        try:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 0.5)
            return mesh.get_volume()
        except:
            points = np.asarray(pcd.points)
            return np.prod(np.max(points, axis=0) - np.min(points, axis=0))

    def _compute_height_width_ratio(self, points):
        min_bounds = np.min(points, axis=0)
        max_bounds = np.max(points, axis=0)
        dimensions = max_bounds - min_bounds
        return dimensions[2] / max(dimensions[0], dimensions[1])

    def _compute_height(self, points):
        """Compute the height (Z dimension) of the rock"""
        min_bounds = np.min(points, axis=0)
        max_bounds = np.max(points, axis=0)
        return max_bounds[2] - min_bounds[2]

    def _compute_cluster_count(self, points, eps=0.1, min_samples=10):
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        return len(np.unique(clustering.labels_))
