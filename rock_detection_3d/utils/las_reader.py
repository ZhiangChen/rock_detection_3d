import laspy
import os
import numpy as np
import json
    
class Read_Las_from_Path(object):
    def __init__(self, las_path):
        self.las_path = las_path
        assert os.path.exists(las_path)
        self._load_las()
        
    def _load_las(self):
        self.las = []
        las_files = [os.path.join(self.las_path, f) for f in os.listdir(self.las_path) if f.endswith('.las')]
        las_files.sort()
        for las_file in las_files:
            self.las.append(laspy.read(las_file))
            
    def __getitem__(self, i):
        return self.las[i]
    
    def __len__(self):
        return len(self.las)
    
    def get_raw(self, i):
        assert i < self.__len__()
        las = self.las[i]
        pos = np.array((las.x.scaled_array(), las.y.scaled_array(), las.z.scaled_array())).transpose().astype(np.float64)
        color = np.array((las.red, las.green, las.blue)).transpose().astype(np.float64) / (2**16)
        try:
            label = (las.isPBR==0)*1
        except AttributeError:
            label = np.logical_not((las.notPBR==1))*1
        
        return pos, color, label
    
    def get_normalized(self, i):
        pos, color, label = self.get_raw(i)
        center = pos.mean(axis=0)
        pos = pos - center
        scale = ((1 / np.abs(pos).max()) * 0.999999).reshape((1))
        pos = pos * scale
        
        return pos, color, label

class Read_Las_from_Json(object):
    def __init__(self, json_file):
        self.json_file = json_file
        assert os.path.isfile(json_file)
        self.json_path = os.path.dirname(os.path.abspath(json_file))
        self._load_las()
        
    def _load_las(self):
        self.las = []
        with open(self.json_file, "r") as f:
                filenames = json.load(f)
        las_files = [os.path.join(self.json_path, f) for f in filenames]
        las_files.sort()
        for las_file in las_files:
            self.las.append(laspy.read(las_file))
            
    def __getitem__(self, i):
        return self.las[i]
    
    def __len__(self):
        return len(self.las)
    
    def get_raw(self, i):
        assert i < self.__len__()
        las = self.las[i]
        pos = np.array((las.x.scaled_array(), las.y.scaled_array(), las.z.scaled_array())).transpose().astype(np.float64)
        color = np.array((las.red, las.green, las.blue)).transpose().astype(np.float64) / (2**16)
        try:
            label = (las.isPBR==0)*1
        except AttributeError:
            label = np.logical_not((las.notPBR==1))*1
        
        return pos, color, label
    
    def get_normalized(self, i):
        pos, color, label = self.get_raw(i)
        center = pos.mean(axis=0)
        pos = pos - center
        scale = ((1 / np.abs(pos).max()) * 0.999999).reshape((1))
        pos = pos * scale
        
        return pos, color, label
    
    
las_path = '../../notebooks/data/rocklas/prediction'