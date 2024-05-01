# Unsupervised 3D Rock Segmentation

## Overview

This module extends the 3D Rock Detection project by focusing on unsupervised segmentation techniques for 3D rock point clouds. It emphasizes k-means clustering and region growing segmentation to enhance rock detection accuracy. The k-means clustering technique groups similar points based on their spatial features, while region growing relies on seed points and neighboring criteria to expand regions. These techniques enable precise segmentation of rock points from supporting surfaces, thus facilitating in-depth geometric analysis of geological formations.

## Workflow

The module involves two primary unsupervised segmentation techniques:

### K-Means Segmentation

1. Initialization: Configure KMeansSegmentation with the point cloud and clustering parameters.
2. Feature Extraction: Extract features like coordinates, height, slope, and normals, then normalize and weight them.
3. K-Means Clustering: Apply the k-means algorithm to classify points into distinct clusters.

Results: <br>
<img src="./images/Kmeans1.png" height="200"> <img src="./images/Kmeans2.png" height="200">

### Region Growing Segmentation

1. Initialization: Set up RegionGrowingSegmentation with the point cloud, neighborhood size, and smoothness threshold (between [-1,1]).
2. Seed Selection: Identify terrain and rock seeds based on point heights.
3. Region Growing: Expand regions from seeds by evaluating neighboring point similarities. Points are added to a region if their normal vectors have a dot product greater than the smoothness threshold with the seed point's normal vector.

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

<!-- ### Running Region Growing Segmentation:
1. Load the point cloud data using open3d as an open3d geometry object.
2. Use RegionGrowingSegmentation with appropriate parameters, then perform segmentation using the segment method.
3. Assess the regions' boundaries by visualizing the segmented point clouds.
 -->

<!-- 
## Todo
-  -->