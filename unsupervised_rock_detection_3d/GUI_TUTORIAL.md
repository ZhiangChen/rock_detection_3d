# 3D Rock Segmentation Tool V2 - Comprehensive User Guide

## Table of Contents
1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Workflow Stages](#workflow-stages)
4. [Configuration System](#configuration-system)
5. [Detailed Panel Guide](#detailed-panel-guide)
---

## Overview

The 3D Rock Segmentation Tool V2 is a PyQt5-based GUI application designed to segment rocks from pedestals in 3D point cloud data, reconstruct meshes, and perform geometric analysis. The workflow is organized into sequential stages:

**Data Loading & Preprocessing → Seed Selection → Interface Constraint → Segmentation → Mesh Reconstruction → Geometric Analysis**

---


## Getting Started

### Launching the Application

```bash
python RegionGrowing_GUI_refactored.py
```

### First-Time Setup

1. **Select User**: Choose your name from the dropdown at the top of the window
2. **Configuration**: The app automatically loads settings from `config.yaml` (see [Configuration System](#configuration-system))
3. **Workspace Layout**: 
   - Left column: Data panel, Segmentation controls.
   - Right column: Interface panel, Mesh panel, Workflow progress indicators
   - Bottom: Status message and export file paths (You can hover over the file path text to see the full file paths)

### UI Elements

- **Current File**: Shows the currently loaded point cloud file
- **EPSG**: Displays the coordinate reference system code
- **Workflow Progress**: Real-time checklist showing completed stages

---

## Workflow Stages

The tool follows a sequential workflow with seven main stages:

### Stage 1: Data Loading & Preprocessing
- Load point cloud (.las/.laz) or database (CSV for batch processing)
- Automatic preprocessing filters are applied (SOR, vertical noise filtering)

### Stage 2: Seed Selection
- Identify initial rock and pedestal seed points
- Choose between automatic detection or manual selection

### Stage 3: Interface Constraint (Optional)
- Define the rock-pedestal boundary for improved segmentation and mesh reconstruction
- Single-line or multi-part interface options

### Stage 4: Region Growing Segmentation
- Segment rock from pedestal using configurable thresholds
- Iteratively refine results by adjusting smoothness, curvature, and proximity parameters

### Stage 5: Mesh Reconstruction
- Interpolate missing faces (bottom face preparation)
- Remove noise (DBSCAN and SOR filtering)
- Compute normals and generate watertight mesh via Poisson reconstruction

### Stage 6: Geometric Analysis
- Calculate dimensional properties (height, width, length)
- Compute orientation angles (alpha, beta)
- Export results to CSV

**Note**: See [Detailed Panel Guide](#detailed-panel-guide) for comprehensive explanations of each option and parameter.

---

## Configuration System

### Config File Hierarchy

The application searches for `config.yaml` in this order:

1. **Environment variable**: `ROCK3D_CONFIG=/path/to/config.yaml`
2. **Current directory**: Walks up from working directory and script location
3. **User config**: `~/.config/rock3d/config.yaml`
4. **Fallback**: Built-in `DEFAULT_CONFIG` in the script

### Key Configuration Sections

#### Users
```yaml
users:
  - Deep Rodge
  - Zhiang Chen
  - Ramon Arrowsmith
```
**Purpose**: Populates the user dropdown menu

#### Thresholds
```yaml
thresholds:
  smoothness: 0.9        # Higher = smoother transitions (0.0-1.0)
  curvature: 0.1         # Lower = less curvature tolerance (0.0-1.0)
  basal_proximity: 0.05  # Distance threshold in meters
```

#### Filters
```yaml
filters:
  sor: true                      # Enable Statistical Outlier Removal
  vertical: true                 # Enable vertical noise filtering
  k_neighbors: 10                # Base neighbor count for SOR
  std_ratio: 2.0                 # Standard deviation multiplier
  adaptive_k_neighbors: true     # Auto-adjust k based on point density
  vertical_std: 1.0              # Z-score cutoff for vertical filtering
  cluster_cleanup: false         # Enable DBSCAN cluster pruning
  cluster_eps: 0.03              # DBSCAN radius in meters
  cluster_dbscan_min_points: 20  # Minimum cluster size
  cluster_min_pct: 0.01          # Keep clusters with ≥1% of points
  basal_clipping: true           # Clip rock against basal surface
  basal_clip_threshold: 0.0      # Clipping tolerance in meters
```

**Adaptive K-Neighbors**:
- When `true`: Uses coefficient of variation (CV) to determine k
  - CV < 0.22 (uniform spacing, e.g., UAV SfM): k=30
  - CV > 0.22 (variable spacing, e.g., UAV LiDAR): k=10
- When `false`: Uses fixed `k_neighbors` value

#### Normal Computation
```yaml
normals:
  method: PyMeshLab  # PyMeshLab | Open3D
  k: 200             # Number of neighbors for normal estimation
```

**Method Comparison**:
- **PyMeshLab**: More smoother normals, handles complex geometries better (recommended for better mesh reconstruction)
- **Open3D**: Use as fallback if PyMeshLab method is not yielding satisfying results (Mostly for rocks that are very flat).

#### Output Paths
```yaml
paths:
  pcd_dir: "{input_dir}"   # Save segmented point clouds here
  mesh_dir: "{input_dir}"  # Save meshes here
  csv_dir: "{input_dir}"   # Save analysis CSVs here
```

**Placeholders**:
- `{input_dir}`: Directory of the input file
- `{pbr}`: Name of the current PBR
- `{ts}`: Timestamp (YYYYMMDD_HHMMSS)

---

## Detailed Panel Guide

### Dataset and Preprocessing Panel

#### Load Point Cloud
**Purpose**: Load a single .las/.laz file for processing

**Steps**:
1. Click "Load Point Cloud"
2. Navigate to your point cloud file
3. Select a .las or .laz file
4. The point cloud will be visualized in a new window
5. Navigate in the 3D visualizer using:
    - **Left click drag**: Rotate the point cloud in any angle.
    - **Scroll**: Zoom in and out
    - **Shift + Left click drag**: Rotate the point cloud in the camera view plane.
    - **Ctrl + Left click drag**: Move the point cloud around in 3D space.
    - **"+" or "-" keyboard keys**: Increase or decrease point size.
    - **"]" or "[" keyboard keys**: Increase or decrease focal length of camera.

**What Happens**:
- File is loaded and centered (mean subtraction)
- EPSG code is extracted (if available)
- RGB colors are loaded (if available)
- Output folder is set to input file's directory


#### Load Database
**Purpose**: Enable batch processing of multiple rocks

**Database Format** (CSV):
```csv
pbr_name,pbr_location,false_positive,processed,...
pbr1,/path/to/pbr1.las,false,false,...
pbr2,/path/to/pbr2.las,false,false,...
```

**Steps**:
1. Click "Load Database"
2. Select your database CSV file
3. The first unprocessed entry will be loaded automatically

**What Happens**:
- Database is loaded into memory
- First row where `processed=false` and `false_positive=false` is selected
- Corresponding point cloud is loaded
- "Load Next PBR" button becomes available

**Tips**:
- Use `generate_pbr_database.py` to create databases from folders
- Mark false positives during processing to skip them later
- Database tracks segmentation and mesh paths automatically

#### Mark False Positive / Log False Positive
**Purpose**: Flag rocks that shouldn't be processed (e.g., not actual rocks or rocks with too much missing data)

**Steps**:
1. Load a point cloud or database entry
2. Click "Log False Positive"
3. File is moved to `false_positives/` subfolder
4. Database entry is updated (if using database mode)

**When to Use**:
- Point cloud contains vegetation, not a rock
- Rock is too fragmented for analysis
- Incorrect extraction from larger dataset


---

### Interface Constraint Workflow Panel

#### Preview Auto Seeds
**Purpose**: Automatically detect rock and pedestal seed points

**Algorithm**:
1. **Rock Seed**: Highest point adjusted for distance from center
2. **Pedestal Seed**: Lowest point in the point cloud

**Preprocessing Applied**:
- Statistical Outlier Removal (SOR)
- Vertical noise filtering (removes high-variance vertical columns)
- Point cloud colors set to uniform gray for clarity

**Visualization**:
- Gray point cloud
- Red point: Rock seed
- Blue point: Pedestal seed

**What Happens Next**:
- Seeds are stored in memory
- Interface constraint buttons become enabled
- "Run Region Growing" becomes available

**Tips**:
- Works best on well-centered, clean point clouds
- Check that red seed is on the rock, blue seed on pedestal
- If seeds are wrong, use manual selection instead
- Auto seeds work well for simple, convex rocks

#### Start Manual Seed Selection (Recommended)
**Purpose**: Manually pick seed points for more control

**Steps**:
1. Click "Start Manual Seed Selection"
2. Dialog appears: "Step 1/2: Select rock seeds"
3. In the 3D viewer:
   - **Shift + Left Click**: Add a seed point
   - **Shift + Right Click**: Undo last point
4. Click "Next" when rock seeds are complete
5. Dialog updates: "Step 2/2: Select pedestal seeds"
6. Select pedestal seeds using same controls
7. Click "Done" to finalize

**Best Practices**:
- **Rock Seeds**: Pick 5-10 points clearly on the rock surface
- **Pedestal Seeds**: Pick 5-10 points clearly on the pedestal
- More seeds = more stable segmentation
- Avoid boundary points (pick interior points)
- Distribute seeds across the region

#### Interface Constraint Tools... (Dialog)

#### Single Complete Interface Input
**Purpose**: Define the rock-pedestal boundary with a single line

**When to Use**:
- Clear, continuous interface boundary
- Rock sits on a relatively flat pedestal
- Single contact surface between rock and pedestal

**Steps**:
1. Click "Single Complete Interface Input"
2. 3D viewer opens with point picking enabled
3. Pick 5-20 points following the interface curve in order.
4. Click "Done" when complete

**What Happens**:
- Algorithm densifies the curve between picked points
- interface constraint points are computed
- Point cloud is colored:
  - Gray: Regular points
  - Red: Interface constraint points
- "Run Region Growing" and "Mesh Workflow" buttons enable

**Tips**:
- 8-12 points usually sufficient

#### Multi-Part Interface Input
**Purpose**: Handle complex interfaces with multiple segments or multiple supports

- **Interface Part**: Rock-pedestal contact surface (actual interface)
- **Lateral Part**: Side surface of rock (not touching pedestal)

**When to Use**:
- Interface has distinct segments (e.g., front, side, back)
- Some segments are lateral supports (not interface contact)
- Complex or irregular interface geometry
- Need different treatment for different segments

**Steps**:

**Part 1: Specify Number of Parts**
1. Click "Multi-Part Interface Input"
2. Dialog appears: "Specify the number of interface constraint parts"
3. Choose number of parts (2-6) based on how many interface parts you want to input.
4. Click "Start Selection"

**Part 2: Define Each Part**
For each part (e.g., Part 1 of 3):
1. Dialog shows: "Collect points for part X of Y"
2. Pick points in the 3D viewer, in order.
3. Check "Part X is lateral" if this segment is for lateral support. This is important as then these points would not be used for the alpha angle analysis.
4. Click "Save Part" (or "Finish" for last part)
5. Repeat for each part

**Part 3: Close Loop Option**
- On the final part, checkbox appears: "Close loop on final part"
- **Checked** (default): Algorithm connects last part back to first part
- **Unchecked**: Interface remains open-ended

**Visualization**:
- Each part gets a different color from the palette
- Colors: Red, Green, Blue, Yellow, Magenta (cyclical)

**Example Use Case**:
1. **Three-Part Rock**:
   - Part 1: Front interface (not lateral)
   - Part 2: Side of rock (lateral)
   - Part 3: Back interface (not lateral)
   - Close loop: Yes


**Tips**:
- Plan your parts before starting
- Lateral parts help constrain the mesh and interpolate missing faces but aren't basal contacts
- Close loop unless you have a specific reason not to
- More parts = more control but more time
- Use consistent direction (clockwise or counter-clockwise)
- The parts should be input in the order like they are in space, as the line interpolation uses last point of previous part and the first point of current part to connect and interpolate the line between them.

---

### Segmentation Controls Panel

This panel contains the core region growing algorithm parameters.

#### Smoothness Threshold (0.0 - 1.0)
**Default**: 0.9

**What It Means**:
- Measures how similar neighboring normals are or how smooth a point is based on the normals of neighboring points
- Higher values = require more similar normals to group points
- Lower values = allow more dissimilar normals

**Technical**: Minimum dot product between point normal and its neighbors' normals

**When to Adjust**:
- **Increase** (0.90-0.99): More sensitive stopping criteria. Higher smoothness required to continue/expand segmentation, helps avoid over-segmentation.
- **Decrease** (0.80-0.88): Less sensitive stopping criteria. Useful when default parameters lead to under-segmented rock region (rock not fully captured).

**Visual Effect**:
- Too high: Segmentation stops too early, rock incomplete
- Too low: Segmentation bleeds into pedestal

**Tips**:
- Start with default (0.9)
- Adjust in small increments (0.02)
- Monitor segmentation visualization carefully

#### Curvature Threshold (0.0 - 1.0)
**Default**: 0.1

**What It Means**:
- Maximum allowed curvature for point to join region
- Lower values = only low-curvature (flat) areas accepted
- Higher values = high-curvature (curved) areas allowed

**Technical**: Computed from eigenvalue decomposition of local covariance matrix

**When to Adjust**:
- **Decrease** (0.05-0.08): Rock has mostly flat surfaces
- **Increase** (0.15-0.25): Rock has rounded, curved surfaces

**Visual Effect**:
- Too low: Segmentation stops at curved regions
- Too high: Segmentation includes noisy high-curvature areas

**Tips**:
- Smooth, rounded rocks need higher thresholds
- Angular, faceted rocks can use lower thresholds

#### Interface Proximity (0.0 - 1.0)
**Default**: 0.05 (5cm)

**What It Means**:
- Distance threshold (in meters) from basal interface
- Points within this distance are strongly constrained
- Only applies if interface constraint is defined

**Technical**: Uses k-d tree to compute nearest distance to basal points

**When to Adjust**:
- **Increase** (0.08-0.15): Wide transition zone between rock and pedestal
- **Decrease** (0.02-0.04): Sharp, well-defined interface

**Visual Effect**:
- Too high: Rock region expansion stops early; segmentation boundary might not align perfectly with interface.
- Too low: Segmentation may spill into pedestal region near interface.

**Tips**:
- Higher point cloud resolution allows tighter thresholds
- If adjusting smoothness and curvature thresholds still leads to over-segmentation, increase Interface Proximity threshold to stop segmentation earlier and prevent spill into pedestal

#### Run Region Growing
**Purpose**: Execute the segmentation algorithm

**Prerequisites**:
- Point cloud loaded
- Seeds selected (auto or manual)

**What Happens**:
1. Region growing starts from seed points
2. Iteratively adds neighboring points based on thresholds
3. Algorithm processes both rock and pedestal regions simultaneously
4. Remaining unlabeled points are interpolated using Label propagation.
5. Final unlabeled points (label = -1) are assigned to pedestal (label = 0)

**Visualization**:
- **Red**: Rock (label = 1)
- **Blue**: Pedestal (label = 0)

**Process Time**: 5-30 seconds depending on point count and compute available

**What to Check**:
- Rock is completely red
- Pedestal is completely blue
- No red bleeding into pedestal
- No blue patches in rock

**If Results Are Poor**:
1. Adjust thresholds to make the region growing conservative or progessive
2. Re-run region growing (quick iteration)
3. Or re-select seeds to add points in the misclassified regions
4. Or redefine interface constraints

**Tips**:
- Save segmentation only when satisfied and then move on to mesh reconstruction
- Parameters can be adjusted and re-run without reloading

#### Save Segmented Point Cloud
**Purpose**: Export the segmentation results

**When Available**: After successful region growing

**File Naming**: `{pbr_name}_segmented.las`

**Default Location**: Same directory as input file (configurable in config.yaml)

**What Gets Saved**:
- Point coordinates
- RGB colors (red/blue labels)
- Segmentation labels (classification field)
- Basal parts metadata (if multi-part interface used)

**Tips**:
- Always save before proceeding to mesh reconstruction
- Database automatically updates with file path

---

### Mesh and Analysis Panel

#### Mesh Workflow... (Dialog)
**Purpose**: Opens a dedicated dialog for mesh reconstruction workflow

**Workflow in Dialog**:
1. Prepare missing faces using the interface constraint points
2. Remove DBSCAN floating noise (iterative, optional)
3. Remove SOR noise (iterative, optional)
4. Compute normals
5. Complete reconstruction
6. Save mesh

#### Interpolate Missing Faces
**Purpose**: Prepare the missing faces of the rock mesh that were blocked by pedestal or lateral supports, as they are needed for mesh reconstruction.

**What It Does**:
- Separates rock points from pedestal points
- Generates interpolated missing faces using NURBS surface
- Clips off extra rock points beyond the generated faces that might interfere with mesh reconstruction (if enabled in config)
- Creates visualization-ready point cloud

**NURBS Parameters** (hard-coded):
- Degree U: 4, Degree V: 4
- Control points: 5x5 grid

**Visualization**:
- Rock points: Red
- Bottom face points: Green
- Combined view with normals

**Tips**:
- This is required before normal computation
- Can be time-consuming for large point clouds (30-60s)
- Check that bottom face looks reasonable; if not, reselect the interface constraint points, as the interpolated faces are directly dependent on them.

**Basal Clipping**:
If `basal_clipping: true` in config:
- Rock points below/beyond the interpolated surface are removed
- `basal_clip_threshold` controls tolerance:
  - 0.0 = exact surface (no penetration)
  - 0.02 = allow 2cm below surface

#### DBSCAN Floating Noise Removal
**Purpose**: Remove disconnected floating noise clusters that might be caused by errors in scanning.

**Location**: Mesh Workflow dialog

**Algorithm**: DBSCAN clustering
- **eps (m)**: Maximum distance between points in a cluster
- **min_samples**: Minimum cluster size (from config: `cluster_dbscan_min_points`)

**Default eps**: 0.02m (from config: `cluster_eps`)

**How It Works**:
1. DBSCAN groups points into clusters
2. Small clusters (< min_samples) = noise
4. Small disconnected pieces are removed

**Steps**:
1. Adjust eps if needed (smaller = more aggressive)
2. Click "Remove Floating Noise"
3. Check visualization
4. Click "Undo Floating Noise" if too aggressive
5. Can repeat with different eps values

**Tips**:
- Start with default (0.02m = 2cm)
- Smaller eps creates more, smaller clusters, leading to more clusters that are < min_samples, meaning more aggressive removal.
- Use before SOR for best results.

#### SOR Noise Removal (in Mesh Workflow)
**Purpose**: Statistical outlier removal from prepared mesh to remove noise overall in the point cloud.

**Parameters** (from config):
- `sor_neighbors`: k-neighbors for density estimation
- `sor_std_ratio`: Standard deviation multiplier

**Adaptive Behavior**:
If `adaptive_k_neighbors: true`:
- Automatically adjusts k based on point spacing
- Uses coefficient of variation (CV) of nearest distances

**Iterative Use**:
- Can be applied multiple times
- Each application saved to history
- "Undo Noise" reverses last application

#### Compute Normals
**Purpose**: Calculate surface normal vectors for reconstruction

**Parameters**:
1. **k (3-500)**: Number of neighbors for normal estimation
   - Lower k: More sensitive to local detail
   - Higher k: Smoother, more stable normals
   - Default: 200 (from config, suggested)

2. **Method**: PyMeshLab or Open3D
   - **PyMeshLab** (recommended):
     - More robust normal orientation
     - Better handles complex geometries and generated smoother normals
   - **Open3D** (fallback):
     - Computes normal for rock points and interpolated missing faces separately and orientated to face outside.
     - The normals may not be close to real as they are computed separately for surfaces, but is a no fail method.


**Tips**:
- k=200 works well for most cases
- Increase k for noisy point clouds
- Decrease k for detailed features
- If major chunk of normals are still facing inwards even after increasing changing k for PyMeshLab, then use Open3D method.
- This step is required before reconstruction

#### Complete Reconstruction
**Purpose**: Generate the final mesh that is required for the geometric analysis

**Algorithm**: Poisson Surface Reconstruction

**Parameters** (hard-coded):
- Depth: 8 (octree depth)
- Scale: 1.1 (slightly larger than point cloud, beyond this, the mesh is cut/clipped)
- Linear fit: False

**What Happens**:
1. Poisson reconstruction creates initial mesh
2. Mesh is cleaned:
   - Remove disconnected components
   - Keep only largest component
3. Small triangles removed (< 0.1% of max edge length)
4. Final mesh stored in `self.mesh_processor.reconstructed_mesh`

**Output**:
- Triangle mesh
- Vertices with positions
- Faces (triangles)

**Note**: If the pointcloud quality is low, there are high chances the mesh reconstruction might fail in the first try. Retry the mesh reconstruction 2-3 times, and if it still fails, then the point cloud has issues, so remove noise or recalculate normals to clean the point cloud.

**Tips**:
- Ensure normals are computed first
- Remove noise thoroughly before this step
- Mesh quality depends on all previous steps

#### Save Mesh
**Purpose**: Export the reconstructed mesh to file

**Output Format**: PLY file
- Binary encoding (compact)
- Vertices (x, y, z)
- Faces (triangle indices)
- Vertex colors (if available)

**File Naming**: `{pbr_name}_mesh.ply`

**Default Location**: Same directory as input file

**Tips**:
- Save before geometric analysis (not strictly required but good practice)

#### Compute Geometric Analysis
**Purpose**: Calculate dimensional and orientation properties after mesh reconstruction

**Computed Properties**:

1. **Dimensions** (PCA-based):
   - Height (Y-axis, second principal component)
   - Width (minimum of X/Z dimensions)
   - Length (maximum of X/Z dimensions)

2. **Ratios**:
   - Height/Width ratio
   - Length/Width ratio

3. **Orientation Vectors**:
   - Major orientations (3 PCA axes)
   - Height-Width face normal
   - Length-Width face normal

4. **Center of Mass**:
   - Volume-weighted centroid
   - Computed from triangle volumes

5. **Angles**:
   - Alpha angle (α): Minimum angle between gravity and vector from CoM to interface points (basal contact points)
   - Alpha rectangular: Tan inverse of width/ height
   - Beta angle (β): Angle between gravity vector and pedestal plane's normal.

**Output**:
1. **CSV File**: Results appended to analysis CSV
2. **Popup Dialog**: Summary of key measurements
3. **Optional Alpha View**: 3D visualization of orientation

**Alpha View Visualization**:
- Shows mesh with pedestal points and CoM and Alpha point markers

#### Load Next PBR
**Purpose**: Process the next entry in the database

**When Available**: After Analysis of current PBR is completed

**What It Does**:
1. Finds next unprocessed entry:
   - `processed == false`
   - `false_positive == false`
2. Loads the corresponding point cloud
3. Resets workflow to start
4. Updates current file display

**Tips**:
- Database state is saved automatically
- Mark false positives to skip them

#### Restart Workflow
**Purpose**: Reset the application to initial state

**What Gets Reset**:
- All point cloud data cleared
- All segmentation results cleared
- All mesh data cleared
- Seeds and interface constraints cleared
- Workflow progress indicators reset
- File paths cleared

**What Persists**:
- User selection
- Configuration settings
- Database connection (if loaded)

**Use Cases**:
- Start over with same file
- Recover from errors
- Free memory

**Tips**:
- Doesn't close visualization windows (manual close)
- Doesn't save any data (manual save first)
- Quick way to start fresh

---

## Keyboard Shortcuts

**3D Visualization** (Open3D window):
- **Left click drag**: Rotate the point cloud in any angle.
- **Scroll**: Zoom in and out
- **Shift + Left click drag**: Rotate the point cloud in the camera view plane.
- **Ctrl + Left click drag**: Move the point cloud around in 3D space.
- **"+" or "-" keyboard keys**: Increase or decrease point size.
- **"]" or "[" keyboard keys**: Increase or decrease focal length of camera.

**Point Picking Mode** (when selecting seeds or interface points):
- **Shift + Left click**: Pick/add a point
- **Shift + Right click**: Undo last picked point

**Main Window**:
- No keyboard shortcuts currently implemented
- Use Tab to navigate between controls

---

## File Formats

### Input Formats
- **.las**: ASPRS LiDAR format (preferred)
- **.laz**: Compressed LAS format

**Required Fields**:
- X, Y, Z coordinates
- Optional: RGB colors, classification, EPSG code

### Output Formats
- **Segmented PCD**: .las file with classification labels
- **Mesh**: .ply file (binary)
- **Analysis**: .csv file with geometric properties

---

**Version**: 2.0  
**Last Updated**: December 2025  
