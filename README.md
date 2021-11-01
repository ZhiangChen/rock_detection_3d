# 3D Rock Detection


## Data
We need the following data for the 3D rock detection. The first two, orthomosaic and mesh models, are obtained from Structure-from-Motion software (e.g., Agisoft). They should have a coordinate reference system of WGS 84 with UTM projection. UTM zones can be found here: https://mangomap.com/robertyoung/maps/69585/what-utm-zone-am-i-in-#. The third data, point cloud, is subsampled from the mesh model. The subsampling can be done in CloudCompare. 
1. Orthomosaic: .tif with WGS 84 and UTM zone
2. Mesh (with texture): .obj with WGS 84 and UTM zone
3. Point cloud: .las, subsampled from .obj

## Workflow
The workflow is described as the following sequence:
1. Rock detection in orthomosaic => bounding boxes
2. Using the detected bounding boxes to crop points in the point cloud => pbr pointcloud candidates
3. Classifying the pbr pointcloud candidates => pbr pointclouds
4. Segmenting the pbr pointclouds => segmented pbr pointclouds  

The objective of the third step is to reduce false detections from the first step (2D detection). 

## Ellipsoid fitting
