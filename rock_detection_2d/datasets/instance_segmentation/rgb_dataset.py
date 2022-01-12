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


class Dataset(object):
    def __init__(self, data_path, pixel_size, transforms=None, include_name=True):
        assert os.path.exists(data_path)
        self.data_path = data_path
        self.transforms = transforms
        self.data_files = [f for f in os.listdir(data_path) if f.endswith(".tif")]
        self.include_name = include_name
        self.pixel_size = pixel_size

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_path, self.data_files[idx])
        
        image = np.asarray(Image.open(data_path).resize((self.pixel_size, self.pixel_size)))
        if image.shape[2] == 4:
            image = image[:,:,:3]
            
        
        shp_path = data_path[:-3]+'shp'
        if not os.path.isfile(shp_path):
            return image, None
        
        shp = gpd.read_file(shp_path)
        masks = []
        for poly in shp.geometry:
            tif = rasterio.open(data_path)
            p = self._poly_from_utm(poly, tif.meta['transform'])
            mask = rasterize([p], (self.pixel_size, self.pixel_size))
            if np.count_nonzero(mask) > 0:
                masks.append(mask)
        
        num_objs = len(masks)
        if num_objs > 0:
            masks = np.stack(masks)
        else:
            return image, None
    
        boxes = []
        
        for i in range(num_objs):
            pos = np.where(masks[i, :, :])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        obj_ids = np.ones(num_objs)
        labels = torch.as_tensor(obj_ids, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

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
        rgb = image[:, :, :3].astype(np.uint8)
        rgb = Image.fromarray(rgb)
        rgb.show()
        masks = target["masks"]
        masks = masks.permute((1, 2, 0))
        masks = masks.numpy()
        masks = masks.max(axis=2) * 255
        masks = Image.fromarray(masks)
        masks.show()

    def imageStat(self, Nm):
        if Nm > len(self.data_files):
            Nm = len(self.data_files)
        images = np.empty((0, 3), float)
        for data_file in self.data_files[: Nm]:
            data_path = os.path.join(self.data_path, data_file)
            image = np.asarray(Image.open(data_path).resize((self.pixel_size, self.pixel_size)))
            if image.shape[2] == 4:
                image = image[:,:,:3]
            image = image.astype(float).reshape(-1, 3)/255.0
            images = np.append(images, image, axis=0)
        return np.mean(images, axis=0).tolist(), np.std(images, axis=0).tolist(), \
               np.max(images, axis=0).tolist(), np.min(images, axis=0).tolist()





if __name__  ==  "__main__":
    #ds = Dataset("./datasets/Rock/data/")
    ds = Dataset("./datasets/iros/bishop/aug/",input_channel=4)
    image_mean, image_std, image_max, image_min = ds.imageStat()
    image, target = ds[0]
    print(image.shape)
    print(image_mean, image_std, image_max, image_min)
