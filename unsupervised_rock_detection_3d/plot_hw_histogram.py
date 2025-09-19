#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_toppling_acceleration(csv_path: str, output_path: str = None, bin_width: float = 1):
    """
    Generate histogram of toppling acceleration.
    
    Args:
        csv_path: Path to the CSV file containing analysis results
        output_path: Optional path to save the plot
        bin_width: Width of histogram bins
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Calculate toppling acceleration
    alpha_rectangular = np.radians(df['alpha_rectangular'])  # Convert to radians
    toppling_acceleration = 1.3 * np.tan(alpha_rectangular)
    
    # Calculate bin edges
    max_acceleration = toppling_acceleration.max()
    min_acceleration = toppling_acceleration.min()
    bins = np.arange(min_acceleration - bin_width, max_acceleration + 2 * bin_width, bin_width)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Create histogram with horizontal orientation
    plt.hist(toppling_acceleration, bins=bins, orientation='horizontal', edgecolor='black', linewidth=1)
    
    # Customize plot
    plt.ylabel('Toppling Acceleration')
    plt.xlabel('Frequency')
    plt.title(f'Distribution of Toppling Acceleration, 1.3*tan(alpha) (N={len(toppling_acceleration)})')
    plt.grid(True, alpha=0.3)
    
    # Set more dense y-axis ticks
    plt.locator_params(axis='y', nbins=20)  # Increase the number of bins
    
    # Add statistical information
    mean_acceleration = toppling_acceleration.mean()
    median_acceleration = toppling_acceleration.median()
    std_acceleration = toppling_acceleration.std()
    
    stats_text = f'Mean: {mean_acceleration:.2f}\nMedian: {median_acceleration:.2f}\nStd: {std_acceleration:.2f}'
    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save or show plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate histogram of toppling acceleration')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file with analysis results')
    parser.add_argument('--output', type=str, help='Path to save the plot (optional)')
    parser.add_argument('--bin-width', type=float, default=1, help='Width of histogram bins (default: 1)')
    
    args = parser.parse_args()
    
    plot_toppling_acceleration(args.csv_file, args.output, args.bin_width)

if __name__ == "__main__":
    main()
