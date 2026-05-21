import os
import laspy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd  # Add this import at the top

def get_rock_dimensions(las_path, padding=0.2):
    # Read LAS file
    las = laspy.read(las_path)
    
    # Get points
    points = np.vstack((las.x, las.y, las.z)).transpose()
    
    # Calculate bounding box
    min_bounds = np.min(points, axis=0)
    max_bounds = np.max(points, axis=0)
    
    # Remove padding from x and y dimensions
    x_size = max_bounds[0] - min_bounds[0] - (2 * padding)
    y_size = max_bounds[1] - min_bounds[1] - (2 * padding)
    height = max_bounds[2] - min_bounds[2]
    
    return x_size, y_size, height

def analyze_rocks(folder_path, padding=0.2):
    # Get only LAZ files
    las_files = [f for f in os.listdir(folder_path) if f.endswith('.laz')]
    
    if not las_files:
        print("No .laz files found in the specified folder!")
        return None, None, None
    
    dimensions = []
    ratios = []
    rock_names = []  # Add list for rock names
    
    print("Analyzing rock dimensions...")
    for las_file in tqdm(las_files):
        file_path = os.path.join(folder_path, las_file)
        try:
            x_size, y_size, height = get_rock_dimensions(file_path, padding)
            width = max(x_size, y_size)  # Use the larger horizontal dimension as width
            
            if width > 0:  # Avoid division by zero
                hw_ratio = height / width
                dimensions.append((width, height))
                ratios.append(hw_ratio)
                rock_names.append(las_file.replace('.laz', ''))  # Store rock name without extension
        except Exception as e:
            print(f"Error processing {las_file}: {str(e)}")
    
    # Create DataFrame with measurements
    rock_data = pd.DataFrame({
        'rock_name': rock_names,
        'width': [d[0] for d in dimensions],
        'height': [d[1] for d in dimensions],
        'hw_ratio': ratios
    })
    
    # Calculate percentile and add is_top_5 column
    percentile_95 = np.percentile(ratios, 95)
    rock_data['is_top_5_percent'] = rock_data['hw_ratio'] >= percentile_95
    
    # Save to CSV
    csv_path = os.path.join(folder_path, 'rock_measurements.csv')
    rock_data.to_csv(csv_path, index=False)
    print(f"\nRock measurements saved to: {csv_path}")
    
    # Convert arrays for existing functionality
    ratios = np.array(ratios)
    dimensions = np.array(dimensions)
    top_5_mask = ratios >= percentile_95
    
    # Create visualizations
    plt.figure(figsize=(15, 5))
    
    # Height/Width ratio histogram
    plt.subplot(131)
    plt.hist(ratios, bins=30, edgecolor='black')
    plt.axvline(x=percentile_95, color='r', linestyle='--', 
                label=f'95th percentile: {percentile_95:.2f}')
    plt.xlabel('Height/Width Ratio')
    plt.ylabel('Frequency')
    plt.title('Distribution of Height/Width Ratios')
    plt.legend()
    
    # Width vs Height scatter plot
    plt.subplot(132)
    plt.scatter(dimensions[:, 0], dimensions[:, 1], alpha=0.5, label='All rocks')
    plt.scatter(dimensions[top_5_mask, 0], dimensions[top_5_mask, 1], 
                color='red', alpha=0.7, label='Top 5%')
    plt.xlabel('Width (m)')
    plt.ylabel('Height (m)')
    plt.title('Rock Dimensions')
    plt.legend()
    
    # Top 5% ratio histogram
    plt.subplot(133)
    plt.hist(ratios[top_5_mask], bins=15, edgecolor='black', color='red')
    plt.xlabel('Height/Width Ratio')
    plt.ylabel('Frequency')
    plt.title('Top 5% Height/Width Ratios')
    
    plt.tight_layout()
    plt.savefig('rock_analysis.png')
    plt.show()
    
    # Print statistics
    print("\nStatistics:")
    print(f"Total rocks analyzed: {len(ratios)}")
    print(f"Average H/W ratio: {np.mean(ratios):.2f}")
    print(f"Median H/W ratio: {np.median(ratios):.2f}")
    print(f"95th percentile H/W ratio: {percentile_95:.2f}")
    print(f"Number of rocks in top 5%: {np.sum(top_5_mask)}")
    
    return ratios, dimensions, top_5_mask

if __name__ == "__main__":
    folder_path = '/Users/deeprodge/Downloads/DREAMS/PG&E/rock_detection_3d/unsupervised_rock_detection_2d/box_pbr_test'
    ratios, dimensions, top_5_mask = analyze_rocks(folder_path, padding=0.2)
