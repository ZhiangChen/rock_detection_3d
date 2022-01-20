"""
Zhiang Chen, Jan 2022

"""

import os
import numpy as np
import torch
from PIL import Image
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import geopandas as gpd
from shapely.geometry import Polygon
import rioxarray
import rasterio
from rasterio.features import rasterize
import json
from random import shuffle
import cv2

def create_datasets(data_path, split=(0.6,0.8)):
    tif_files = [f for f in os.listdir(data_path) if f.endswith('.tif')]
    shuffle(tif_files)
    nm = len(tif_files)
    train_files = tif_files[:int(nm*split[0])]
    valid_files = tif_files[int(nm*split[0]):int(nm*split[1])]
    test_files = tif_files[int(nm*split[1]):]
    with open(os.path.join(data_path, 'train_split.json'), 'w') as f:
        json.dump(train_files, f)

    with open(os.path.join(data_path, 'valid_split.json'), 'w') as f:
        json.dump(valid_files, f)

    with open(os.path.join(data_path, 'test_split.json'), 'w') as f:
        json.dump(test_files, f)


class Dataset(object):
    def __init__(self, json_file_list, pixel_size, input_channel=(0,1,2), transforms=None, include_name=True):
        self.data_files = []
        for json_file in json_file_list:
            assert os.path.isfile(json_file)
            data_path = os.path.dirname(os.path.realpath(json_file))
            with open(json_file, "r") as f:
                data_files = json.load(f)
                assert len(data_files) >= 1
                data_abs_path = [os.path.join(data_path, data_file) for data_file in data_files]
                self.data_files += data_abs_path
                
                
        self.transforms = transforms
        self.include_name = include_name
        self.pixel_size = pixel_size
        self.input_channel = input_channel

    def __getitem__(self, idx):
        
        data_path = self.data_files[idx]
        
        image = np.asarray(Image.open(data_path).resize((self.pixel_size, self.pixel_size)))
        image = image[:,:,self.input_channel]
        
        shp_path = data_path[:-3]+'shp'
        if not os.path.isfile(shp_path):
            return image, None
        
        shp = gpd.read_file(shp_path)
        masks = []
        boxes = []
        for poly in shp.geometry:
            if poly is None:
                continue
            tif = rasterio.open(data_path)
            p = self._poly_from_utm(poly, tif.meta['transform'])
            mask = rasterize([p], (tif.height, tif.width))
            #mask = rasterize([p], (self.pixel_size, self.pixel_size))
            mask = cv2.resize(mask, dsize=(self.pixel_size, self.pixel_size), interpolation=cv2.INTER_LINEAR)
            mask = mask > 0
            if np.count_nonzero(mask) > 0:
                pos = np.where(mask)
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                if (xmax - xmin > 0) & (ymax - ymin > 0):
                    masks.append(mask)
                    boxes.append([xmin, ymin, xmax, ymax])
        
        num_objs = len(masks)
        if num_objs > 0:
            masks = np.stack(masks)
        else:
            return image, None
   

        image = image/255.
        image = np.moveaxis(image, 2, 0)
        boxes = np.asarray(boxes)
        obj_ids = np.ones(num_objs)
        labels = np.asarray(obj_ids, dtype=np.int64)
        masks = np.asarray(masks, dtype=np.uint8)
        image_id = np.asarray([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = np.zeros((num_objs,), dtype=np.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.include_name:
            target["image_name"] = data_path

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        return image, target

    def __len__(self):
        return len(self.data_files)
    
    def _poly_from_utm(self, poly, transform):
        poly_pts = []
        for i in np.array(poly.exterior.coords):

            # Convert polygons to the image CRS
            poly_pts.append(~transform * tuple(i))

        # Generate a polygon object
        new_poly = Polygon(poly_pts)
        return new_poly

    def show(self, idx):
        image, target = self.__getitem__(idx)
        if self.transforms is not None:
            image = image.permute(1, 2, 0)
            rgb = (image.numpy()*255).astype(np.uint8)
            rgb = Image.fromarray(rgb)
            rgb.show()
            masks = target["masks"]
            masks = masks.permute(1, 2, 0)
            masks = masks.numpy()
            masks = masks.max(axis=2) * 255
            masks = Image.fromarray(masks)
            masks.show()
        else:
            rgb = (np.moveaxis(image, 0, -1)*255).astype(np.uint8)
            #rgb = rgb[:, :, :3].astype(np.uint8)
            rgb = Image.fromarray(rgb)
            rgb.show()
            masks = target["masks"]
            masks = np.moveaxis(masks, 0, -1)
            masks = masks.max(axis=2) * 255
            masks = Image.fromarray(masks)
            masks.show()

    def imageStat(self, Nm):
        if Nm > len(self.data_files):
            Nm = len(self.data_files)
            
        N = len(self.input_channel)
        images = np.empty((0, N), float)
        for data_path in self.data_files[: Nm]:
            image = np.asarray(Image.open(data_path).resize((self.pixel_size, self.pixel_size)))
            image = image[:,:, self.input_channel]
            image = image.astype(float).reshape(-1, N)/255.0
            images = np.append(images, image, axis=0)
        return np.mean(images, axis=0).tolist(), np.std(images, axis=0).tolist(), \
               np.max(images, axis=0).tolist(), np.min(images, axis=0).tolist()

