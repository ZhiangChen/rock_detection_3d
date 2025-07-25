import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import os

class FilterStats:
    def __init__(self, output_dir="filter_analysis"):
        self.output_dir = output_dir
        self.filter_stats = defaultdict(list)
        self.rejection_stats = defaultdict(int)
        os.makedirs(output_dir, exist_ok=True)
        
    def log_filter(self, name, points_before, points_after, pcd_before=None, pcd_after=None, visualize=False):
        """Log statistics for a filter step"""
        reduction = (points_before - points_after) / points_before * 100
        self.filter_stats[name].append({
            'points_before': points_before,
            'points_after': points_after,
            'reduction': reduction
        })
        
        if visualize and pcd_before is not None and pcd_after is not None:
            self.visualize_filter(name, pcd_before, pcd_after)
    
    def log_rejection(self, reason, rock_name):
        """Log why a rock was rejected"""
        self.rejection_stats[reason] += 1
        
    def visualize_filter(self, name, pcd_before, pcd_after):
        """Visualize before/after of a filter"""
        pcd_before_vis = pcd_before.paint_uniform_color([1, 0, 0])  # Red
        pcd_after_vis = pcd_after.paint_uniform_color([0, 1, 0])   # Green
        o3d.visualization.draw_geometries([pcd_before_vis, pcd_after_vis],
                                        window_name=f"Filter: {name}")
    
    def generate_report(self):
        """Generate analysis report with plots"""
        # 1. Filter reduction statistics
        plt.figure(figsize=(15, 5))
        
        plt.subplot(121)
        avg_reductions = {name: np.mean([s['reduction'] for s in stats]) 
                         for name, stats in self.filter_stats.items()}
        plt.bar(avg_reductions.keys(), avg_reductions.values())
        plt.title('Average Point Reduction by Filter')
        plt.xticks(rotation=45)
        plt.ylabel('Average Reduction %')
        
        plt.subplot(122)
        plt.bar(self.rejection_stats.keys(), self.rejection_stats.values())
        plt.title('Rock Rejection Reasons')
        plt.xticks(rotation=45)
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'filter_analysis.png'))
        
        # Save detailed stats to JSON
        with open(os.path.join(self.output_dir, 'filter_stats.json'), 'w') as f:
            json.dump({
                'filter_stats': dict(self.filter_stats),
                'rejection_stats': dict(self.rejection_stats)
            }, f, indent=2)
