import os
import numpy as np
import torch
import random 
import rasterio
import laspy
import geopandas as gpd
import torchvision.transforms as T
from tqdm import tqdm
from shapely.geometry import box
from rasterio.windows import Window
from torchvision.ops import box_convert
from groundingdino.util.inference import load_model, predict
from pyproj import CRS
from PIL import Image
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse



# ----------------------------
# 1. Orthomosaic Processing
# ----------------------------

def log_time(message, start_time):
    elapsed = time.time() - start_time
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}: {elapsed:.2f} seconds")

class OrthomosaicDataset(torch.utils.data.Dataset):
    def __init__(self, tif_path, window_size=512, blank_threshold=0.8, validate_tiles=True):
        init_start = time.time()
        
        raster_start = time.time()
        self.src = rasterio.open(tif_path)
        
        self.window_size = window_size
        self.blank_threshold = blank_threshold
        self.valid_indices = []
        self.validate_tiles = validate_tiles
        
        transform_start = time.time()
        self.transform = T.Compose([
            T.Resize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        if validate_tiles:
            self._precalculate_valid_indices()
        else:
            print("Skipping tile validation as per configuration.")
            total_tiles = (self.src.height // self.window_size) * (self.src.width // self.window_size)
            self.valid_indices = list(range(total_tiles))
        
        log_time("Total initialization", init_start)

    def _precalculate_valid_indices(self):
        scan_start = time.time()
        total_tiles = (self.src.height // self.window_size) * (self.src.width // self.window_size)
        
        for idx in tqdm(range(total_tiles), desc="Validating tiles"):
            h_idx = idx // (self.src.width // self.window_size)
            w_idx = idx % (self.src.width // self.window_size)
            window = Window(w_idx * self.window_size, 
                            h_idx * self.window_size,
                            self.window_size, 
                            self.window_size)
            
            # Validate window bounds
            if (window.col_off + window.width > self.src.width or
                window.row_off + window.height > self.src.height):
                print(f"Skipping out-of-bounds tile {idx} at (h_idx={h_idx}, w_idx={w_idx})")
                continue

            try:
                tile = self.src.read(window=window)
                if self._is_valid_tile(tile):
                    self.valid_indices.append(idx)
            except rasterio.errors.RasterioIOError as e:
                print(f"Error reading tile {idx} at (h_idx={h_idx}, w_idx={w_idx}): {e}")
                continue
                
        log_time("Total tile validation", scan_start)
        print(f"Found {len(self.valid_indices)} valid tiles ({len(self.valid_indices)/total_tiles*100:.1f}%)")

    def _is_valid_tile(self, tile):
        if tile.shape[0] == 4:
            tile = tile[:3]
            
        tile_sum = np.sum(tile, axis=0)
        valid_pixels = np.sum(tile_sum > 0)
        return (valid_pixels / (self.window_size ** 2)) > (1 - self.blank_threshold)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        h_idx = actual_idx // (self.src.width // self.window_size)
        w_idx = actual_idx % (self.src.width // self.window_size)
        
        window = Window(w_idx * self.window_size,
                        h_idx * self.window_size,
                        self.window_size,
                        self.window_size)
        
        # Validate window bounds
        if (window.col_off + window.width > self.src.width or
            window.row_off + window.height > self.src.height):
            raise ValueError(f"Tile {actual_idx} window is out of bounds: {window}")

        try:
            image = self.src.read(window=window)
        except rasterio.errors.RasterioIOError as e:
            print(f"Error reading tile {actual_idx} at (h_idx={h_idx}, w_idx={w_idx}): {e}")
            raise

        if image.shape[0] == 4:
            image = image[:3]
        image = np.transpose(image, (1, 2, 0))
        image_source = Image.fromarray((image * 255).astype(np.uint8))
        image_transformed = self.transform(image_source)
        
        return image, image_transformed, actual_idx

    def get_epsg(self):
        return self.src.crs.to_epsg()


# ----------------------------
# 2. Rock Detection
# ----------------------------

class RockDetector:
    def __init__(self, model_config, model_weights, device="cpu"):
        init_start = time.time()
        self.model = load_model(model_config, model_weights, device=device)
        
    def detect_rocks(self, dataset, text_prompt, box_thresh=0.15, text_thresh=0.25, max_tiles=None):
        detect_start = time.time()
        
        # Validate all tiles but limit inference to max_tiles
        valid_indices = dataset.valid_indices
        if max_tiles is None or max_tiles > len(valid_indices):
            max_tiles = len(valid_indices)  # Default to all valid tiles if max_tiles is None or exceeds valid tiles
        
        num_tiles = min(max_tiles, len(valid_indices))
        selected_indices = random.sample(valid_indices, num_tiles)
        print(f"Total valid tiles: {len(valid_indices)}")
        print(f"max_tiles: {max_tiles}")
        print(f"Selected {num_tiles} tiles for inference")
        print("Selected_indices:", selected_indices)

        # Initialize results list and counter
        results = []
        total_boxes = 0

        def process_tile(idx, tif_path, window_size, transform, dataset_transform, dataset_crs):
            tile_start = time.time()
            tile_results = []

            # Reopen rasterio dataset in the thread
            with rasterio.open(tif_path) as src:
                h_idx = idx // (src.width // window_size)
                w_idx = idx % (src.width // window_size)
                window = Window(w_idx * window_size, h_idx * window_size, window_size, window_size)

                # Read tile
                try:
                    image = src.read(window=window)
                except rasterio.errors.RasterioIOError as e:
                    print(f"Error reading tile {idx} at (h_idx={h_idx}, w_idx={w_idx}): {e}")
                    return []

                if image.shape[0] == 4:
                    image = image[:3]
                image = np.transpose(image, (1, 2, 0))
                image_source = Image.fromarray((image * 255).astype(np.uint8))
                image_transformed = transform(image_source)

                # Perform inference
                boxes, logits, _ = predict(
                    model=self.model,
                    image=image_transformed,
                    caption=text_prompt,
                    box_threshold=box_thresh,
                    text_threshold=text_thresh,
                    device='cpu'
                )

                if boxes.shape[0] > 0:
                    new_results = self._process_detections(
                        dataset_transform, dataset_crs, idx, boxes, logits, image.shape, h_idx, w_idx, window_size
                    )
                    tile_results.extend(new_results)

            log_time(f"Total tile {idx} processing", tile_start)
            print(f"Found {boxes.shape[0]} potential rocks in tile {idx}")
            return tile_results

        # Use ThreadPoolExecutor for parallel processing
        num_workers = min(4, os.cpu_count() or 1)  # Use up to 4 workers
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks and create a future-to-index mapping
            future_to_idx = {
                executor.submit(
                    process_tile,
                    idx,
                    dataset.src.name,
                    dataset.window_size,
                    dataset.transform,
                    dataset.src.transform,
                    dataset.src.crs,
                ): idx
                for idx in selected_indices
            }
            
            # Process completed tasks as they finish
            for future in tqdm(as_completed(future_to_idx), total=len(selected_indices), desc="Processing tiles"):
                idx = future_to_idx[future]
                try:
                    tile_results = future.result()
                    results.extend(tile_results)
                    total_boxes += len(tile_results)
                except Exception as e:
                    print(f"Error processing tile {idx}: {str(e)}")
                    continue

        log_time("Total rock detection", detect_start)
        print(f"Total rocks detected: {total_boxes}")
        
        if not results:
            print("No rocks detected in the selected tiles.")
            return gpd.GeoDataFrame(columns=["geometry"], crs=dataset.src.crs)
        
        return gpd.GeoDataFrame(results, crs=dataset.src.crs)

    def _process_detections(self, dataset_transform, dataset_crs, idx, boxes, logits, img_shape, h_idx, w_idx, window_size):
        h, w, _ = img_shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        x_offset = w_idx * window_size
        y_offset = h_idx * window_size

        # Vectorized size check
        box_widths = xyxy_boxes[:, 2] - xyxy_boxes[:, 0]
        box_heights = xyxy_boxes[:, 3] - xyxy_boxes[:, 1]
        valid_indices = (box_widths < 0.8 * window_size) & (box_heights < 0.8 * window_size)
        print(f"Found {np.sum(valid_indices)} valid boxes out of {len(valid_indices)}")

        valid_boxes = []
        for i in np.where(valid_indices)[0]:
            bbox = xyxy_boxes[i]  # Rename variable from `box` to `bbox`
            logit = logits[i]

            # Convert to georeferenced coordinates
            x_min, y_min, x_max, y_max = bbox
            geo_xmin = x_min + x_offset
            geo_ymin = y_min + y_offset
            geo_xmax = x_max + x_offset
            geo_ymax = y_max + y_offset

            top_left = dataset_transform * (geo_xmin, geo_ymin)
            bottom_right = dataset_transform * (geo_xmax, geo_ymax)

            valid_boxes.append({
                "geometry": box(top_left[0], bottom_right[1], bottom_right[0], top_left[1]),  # Use `box` function safely
                "confidence": logit.item(),
                "tile_index": idx
            })

        return valid_boxes

# ----------------------------
# 3. Point Cloud Extraction
# ----------------------------

class PointCloudExtractor:
    def __init__(self, laz_path):
        init_start = time.time()
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Initializing PointCloudExtractor...")
        
        load_start = time.time()
        self.laz_path = laz_path
        print(f"Loading point cloud from {laz_path}...")
        self.pc = laspy.read(laz_path)
        log_time("Point cloud loading", load_start)
        
        print(f"Loaded {len(self.pc)} points with CRS: {self.pc.header.parse_crs()}")
        print(f"Point cloud bounds:")
        print(f"X: {self.pc.x.min():.2f} to {self.pc.x.max():.2f}")
        print(f"Y: {self.pc.y.min():.2f} to {self.pc.y.max():.2f}")
        print(f"Z: {self.pc.z.min():.2f} to {self.pc.z.max():.2f}")
        
        # Precompute coordinates during initialization
        prep_start = time.time()
        self.x_coords = self.pc.x.copy()
        self.y_coords = self.pc.y.copy()
        self.min_x = np.min(self.x_coords)
        self.max_x = np.max(self.x_coords)
        self.min_y = np.min(self.y_coords)
        self.max_y = np.max(self.y_coords)
        log_time("Coordinate preprocessing", prep_start)
        
        log_time("Total initialization", init_start)
        
    def extract_rocks(self, geojson_path, output_dir="rock_patches", padding=0.2):
        extract_start = time.time()
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting rock extraction...")
        
        read_start = time.time()
        os.makedirs(output_dir, exist_ok=True)
        bboxes_gdf = gpd.read_file(geojson_path)
        log_time("GeoJSON reading", read_start)
        
        validate_start = time.time()
        self._validate_crs(bboxes_gdf)
        log_time("CRS validation", validate_start)
        
        print(f"\nExtracting {len(bboxes_gdf)} rock patches...")
        for idx, row in tqdm(bboxes_gdf.iterrows(), total=len(bboxes_gdf), desc="Extracting rocks"):
            rock_start = time.time()
            self._process_rock(row, output_dir, padding)
            log_time(f"Rock {idx} extraction", rock_start)
            
        log_time("Total rock extraction", extract_start)

    def _process_rock(self, row, output_dir, padding):
        bounds_start = time.time()
        minx, miny, maxx, maxy = row.geometry.bounds
        print(f"\nProcessing rock {row.name}:")
        print(f"Bounding box (global coordinates): minx={minx}, miny={miny}, maxx={maxx}, maxy={maxy}")
        
        # Calculate padded bounds
        x1 = min(minx, maxx) - padding
        x2 = max(minx, maxx) + padding
        y1 = min(miny, maxy) - padding
        y2 = max(miny, maxy) + padding
        print(f"Padded bounding box: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

        # Bounds check
        if (x2 < self.min_x or x1 > self.max_x or
            y2 < self.min_y or y1 > self.max_y):
            print("🚨 Bounding box completely outside point cloud coverage!")
            return
            
        mask_start = time.time()
        mask = self._create_mask(x1, y1, x2, y2)
        
        if mask.sum() == 0:
            print("⚠️ No points found in search area")
            return

        print(f"Number of points in mask: {mask.sum()}")
        write_start = time.time()
        self._write_patch(mask, os.path.join(output_dir, f"rock_{row.name}.laz"))

    def _create_mask(self, x1, y1, x2, y2):
        # Parallelized mask creation
        def mask_chunk(start, end):
            x_mask = (self.x_coords[start:end] >= x1) & (self.x_coords[start:end] <= x2)
            y_mask = (self.y_coords[start:end] >= y1) & (self.y_coords[start:end] <= y2)
            return x_mask & y_mask

        num_points = len(self.x_coords)
        num_workers = min(4, os.cpu_count() or 1)  # Use up to 4 workers
        chunk_size = num_points // num_workers

        mask_start = time.time()
        mask = np.zeros(num_points, dtype=bool)

        if num_workers > 1:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(mask_chunk, i, min(i + chunk_size, num_points)): i for i in range(0, num_points, chunk_size)}
                for future in as_completed(futures):
                    start_index = futures[future]
                    mask[start_index:start_index + chunk_size] = future.result()
            log_time(f"Parallelized mask creation", mask_start)
        else:
            # Sequential execution
            mask = mask_chunk(0, num_points)
            log_time(f"Sequential mask creation", mask_start)

        points_found = mask.sum()
        print(f"Mask creation completed: {points_found} points found")
        return mask


    def _write_patch(self, mask, output_path):
        """Write the extracted points to a new .las file, preserving global coordinates."""
        filtered_points = self.pc.points[mask].copy()

        # Create a new header and copy relevant attributes
        filtered_header = laspy.LasHeader(point_format=self.pc.header.point_format, version=self.pc.header.version)
        filtered_header.scales = self.pc.header.scales
        filtered_header.offsets = self.pc.header.offsets

        # Set CRS if available
        if self.pc.header.parse_crs():
            filtered_header.parse_crs(self.pc.header.parse_crs())

        # Update header bounds to match the filtered points
        filtered_header.min = [filtered_points['X'].min(), filtered_points['Y'].min(), filtered_points['Z'].min()]
        filtered_header.max = [filtered_points['X'].max(), filtered_points['Y'].max(), filtered_points['Z'].max()]

        # Create a new LAS file with the filtered points
        filtered_pc = laspy.LasData(filtered_header)

        # Assign filtered points directly to the new LAS file
        filtered_pc.points = filtered_points
        
        # Save the filtered point cloud to the specified output path
        filtered_pc.write(output_path)
        print(f"Saved extracted rock patch to {output_path}")

    def _validate_crs(self, gdf):
        pc_crs = self.pc.header.parse_crs().to_epsg()
        if gdf.crs.to_epsg() != pc_crs:
            raise ValueError(f"CRS mismatch: Point cloud {pc_crs} vs boxes {gdf.crs}")

    def calculate_2d_centroid(self, las_file_path):
        """Calculate the 2D centroid (x, y) of a rock from its .las file."""
        las = laspy.read(las_file_path)
        x_coords = las.x
        y_coords = las.y
        centroid_x = np.mean(x_coords)
        centroid_y = np.mean(y_coords)
        return centroid_x, centroid_y


# ----------------------------
# Main Pipeline
# ----------------------------



def validate_epsg_compatibility(tif_path, laz_path):
    """Validate that TIFF and LAS files have the same EPSG code."""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Validating EPSG compatibility...")
    
    # Get TIFF EPSG
    with rasterio.open(tif_path) as src:
        tif_epsg = src.crs.to_epsg()
    
    # Get LAS EPSG
    las = laspy.read(laz_path)
    las_crs = las.header.parse_crs()
    las_epsg = las_crs.to_epsg() if las_crs else None
    
    print(f"TIFF EPSG: {tif_epsg}")
    print(f"LAS EPSG: {las_epsg}")
    
    if tif_epsg != las_epsg:
        print(f"    ERROR: EPSG codes are different!")
        print(f"    TIFF file: EPSG:{tif_epsg}")
        print(f"    LAS file: EPSG:{las_epsg}")
        print("    Cannot proceed with different coordinate reference systems.")
        return False
    
    print(f"✅ EPSG validation passed: Both files use EPSG:{tif_epsg}")
    return True

def run_pipeline(config):
    pipeline_start = time.time()
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting pipeline execution...")
    
    # # Validate EPSG compatibility first
    # if not validate_epsg_compatibility(config["tif_path"], config["laz_path"]):
    #     print("Pipeline stopped due to EPSG mismatch.")
    #     return
    
    if 'dataset' in locals() or 'dataset' in globals():
        del dataset, detector, bboxes_gdf, extractor
        
    dataset = OrthomosaicDataset(config["tif_path"], validate_tiles=config.get("validate_tiles", True))
    
    detector_start = time.time()
    detector = RockDetector(config["model_config"], config["model_weights"])
    log_time("Detector initialization", detector_start)
    
    print(f"TIFF CRS: EPSG:{dataset.get_epsg()}")

    detection_start = time.time()
    bboxes_gdf = detector.detect_rocks(
        dataset,
        text_prompt=config["text_prompt"],
        box_thresh=config["box_thresh"],
        text_thresh=config["text_thresh"],
        max_tiles=config.get("max_tiles")
    )
    log_time("Rock detection", detection_start)
    
    if not bboxes_gdf.empty:
        save_start = time.time()
        bboxes_gdf.to_file(config["output_geojson"], driver="GeoJSON")
        log_time("GeoJSON saving", save_start)
        print(f"Saved {len(bboxes_gdf)} bounding boxes to {config['output_geojson']}")

        extraction_start = time.time()
        extractor = PointCloudExtractor(config["laz_path"])
        extractor.extract_rocks(
            geojson_path=config["output_geojson"],
            output_dir=config["output_dir"],
            padding=config["padding"]
        )
        log_time("Point cloud extraction", extraction_start)
        
    log_time("Total pipeline execution", pipeline_start)
    print(f"\nPipeline completed successfully. Outputs in {config['output_dir']}")

# Configuration
CONFIG = {
    "model_config": "GroundingDINO_files/GroundingDINO_SwinT_OGC.py",
    "model_weights": "GroundingDINO_files/groundingdino_swint_ogc.pth",
    "text_prompt": "isolated rock.",
    "box_thresh": 0.20,
    "text_thresh": 0.25,
    "output_geojson": "pond1_lidar_new_thresholds.geojson",
    "output_dir": "pond1_lidar_new_thresholds",
    "padding": 0.2,
    "max_tiles": 150000,  # To limit processing; for checking
    "validate_tiles": False
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rock Detection Pipeline")
    parser.add_argument("--tif_path", required=True, help="Path to the TIFF orthomosaic file")
    parser.add_argument("--laz_path", required=True, help="Path to the LAZ/LAS point cloud file")
    
    args = parser.parse_args()
    
    # Update CONFIG with command line arguments
    CONFIG["tif_path"] = args.tif_path
    CONFIG["laz_path"] = args.laz_path
    
    run_pipeline(CONFIG)