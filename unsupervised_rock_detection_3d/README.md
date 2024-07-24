# Unsupervised 3D Rock Segmentation

## Overview

This module extends the 3D Rock Detection project by focusing on unsupervised segmentation techniques for 3D rock point clouds. It emphasizes k-means clustering and region growing segmentation to enhance rock detection accuracy. The k-means clustering technique groups similar points based on their spatial features, while region growing relies on seed points and neighboring criteria to expand regions. These techniques enable precise segmentation of rock points from supporting surfaces, thus facilitating in-depth geometric analysis of geological formations.

To enhance usability and enable interactive control, we have also developed a Graphical User Interface (GUI). The GUI allows geologists and students to run segmentation algorithms, visualize each step, and have control over the segmentation by selecting initial seeds or basal points.

**Original and Segmented point cloud:**

<img src="./images/Original.png" height="200"> <img src="./images/RegionGrowing2.png" height="200">

## GUI Features

- **Interactive Segmentation**: Users can select seed points and basal points directly on the point cloud visualization to guide the region growing algorithm.
- **Visualization**: Visualize the original, intermediate, and final segmented point clouds with different colors representing different regions.
- **Feedback Incorporation**: Users can iteratively refine the segmentation results by providing feedback and selecting more basal points.

<img src="./images/GUI_Workflow.png" height="400">

**Interactive Point Selection Visualizer:** <br>
<img src="./images/PointSelection_GUI.png" height="400">

**Completed segmentation in GUI:** <br>
<img src="./images/RegionGrowing_GUI.png" height="500">

## Workflow

The module involves two primary unsupervised segmentation techniques:

### Region Growing Segmentation

1. **Initialization**:

    - Configure the RegionGrowingSegmentation class with the point cloud and segmentation parameters such as voxel size, number of neighbors, smoothness threshold, distance threshold, and curvature threshold.
    - Downsample the point cloud using a voxel grid to reduce computation.
    - Estimate normals for the downsampled point cloud to use in segmentation criteria. Normals are oriented consistently for reliable dot product calculations.
    - Build a KD-Tree for efficient neighbor searches.

2. **Seed Selection**:

    - Compute the bounding box of the point cloud to find the geometric bounds.
    - Determine the centroid of the bounding box in the x and y dimensions.
    - Identify the point with the highest z value near the centroid, representing a potential rock seed.
    - Identify the lowest point in the bounding box, representing a potential terrain seed.

3. **Region Growing**:

    - Precompute neighbors for each point within the specified distance threshold to speed up segmentation.
    - Initialize regions from the selected seed points (highest point for rocks, bottommost point for terrain).
    - Expand regions from seeds by evaluating neighbors based on the segmentation criteria (normal vector smoothness and curvature):
        - Start with the seed point in a queue.
        - For each point, evaluate its neighbors based on the segmentation criteria.
        - Check if neighbors meet the smoothness and curvature thresholds. If they do, add them to the current region and the queue.
        - Assign labels to points based on the region they belong to.
4. **Post Processing**:

    - Perform conditional label propagation to assign labels to remaining unlabeled points based on majority labels of their neighbors.
    - Color the segmented point cloud for visualization by assigning different colors to different regions (e.g., red for rocks, blue for terrain).
    - Visualize the segmented point cloud to assess the regions' boundaries.

#### Mathematical Definitions

- **Smoothness**: Smoothness is evaluated using the dot product between the normal of a point and the normals of its second-order neighbors (neighbors of neighbors). For a point $`( p )`$ with normal $`( \mathbf{n}_p )`$, and its second-order neighbors $`( q )`$ with normals $`( \mathbf{n}_q )`$:

  $$\ Smoothness = \min \left( \mathbf{n}_p \cdot \mathbf{n}_q \right)$$

  where $`\cdot`$ denotes the dot product. We use the minimum value to ensure that we capture the maximum deviation, helping to avoid over-inclusion of points in densely packed clouds.

- **Curvature**: Curvature is computed using the cross product of vectors formed by the difference between neighbor normals and the current normal. For a point $`( p )`$ with normal $`( \mathbf{n}_p )`$, and its neighbors $`( q_i )`$ with normals $`( \mathbf{n}_{q_i})`$:

  $$\
  Curvature = \frac{1}{k} \sum_{i=1}^k \left\| \mathbf{n}_{q_i} \times (\mathbf{n}_{q_i} - \mathbf{n}_p) \right\| $$

  where $`\times`$ denotes the cross product, and $`k`$ is the number of neighbors. This measure captures the variation in normal directions, indicating surface roughness.

#### Usage

- We suggest use of only one segmentation criteria, either smoothness or curvature, not both, depending on the specific requirements of your application.

#### Novelty

One significant novelty in our approach is the use of second-order neighbors for smoothness calculation. Our point cloud is densely packed, resulting in neighboring points being very close to one another. This proximity causes the normal orientation differences to appear nearly parallel, leading almost every point to be included in the region. To address this issue, we modified the approach by considering the minimum dot product (maximum deviation) with second-order neighboring points instead of comparing the immediate neighboring points. This adjustment improves the segmentation accuracy by preventing the over-inclusion of points in a region due to near-parallel normals in densely packed clouds.

Results: <br>
<img src="./images/RegionGrowing1.png" height="200"> <img src="./images/RegionGrowing2.png" height="200">

### K-Means Segmentation

1. Initialization: Configure KMeansSegmentation with the point cloud and clustering parameters.
2. Feature Extraction: Extract features like coordinates, height, slope, and normals, then normalize and weight them.
3. K-Means Clustering: Apply the k-means algorithm to classify points into distinct clusters.

Results: <br>
<img src="./images/Kmeans1.png" height="200"> <img src="./images/Kmeans2.png" height="200">

## Requirements

To use this module, the following packages are required:

1. python >= 3.8
2. scikit-learn: For k-means clustering.
3. open3d: For point cloud processing and normal estimation.
4. numpy: For array operations.
5. matplotlib: For visualization of the segmentation results.

Use the following pip command to install them:

```
pip install scikit-learn open3d numpy matplotlib
```

## Getting Started

### Data Preparation

1. Obtain the point cloud data (in .pcd format).
2. Ensure the data is correctly georeferenced for accurate segmentation.

### Running K-Means Clustering

1. Load the point cloud data using open3d as an open3d geometry object.
2. Create an instance of the KMeansSegmentation class by passing the loaded point cloud, desired number of clusters (typically 2 for rock and bedrock), and feature weights.
3. Use the segment method of the class to perform clustering, and then visualize the segmented results using the visualize_segmentation method.

### Running Region Growing Segmentation

1. Load the point cloud data using open3d as an open3d geometry object.
2. Use RegionGrowingSegmentation with appropriate parameters, then perform segmentation using the segment method.
3. Perform postprocessing using conditional_label_propagation method and color the regions using color_point_cloud method.
4. Assess the regions' boundaries by visualizing the segmented point clouds.

<!-- ### Running Region Growing Segmentation:
1. Load the point cloud data using open3d as an open3d geometry object.
2. Use RegionGrowingSegmentation with appropriate parameters, then perform segmentation using the segment method.
3. Assess the regions' boundaries by visualizing the segmented point clouds.
 -->

<!-- 
## Todo
-  -->