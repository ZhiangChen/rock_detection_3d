import laspy
import os
import numpy as np
import json

class Las_Reader(object):
    def __init__(self, json_file):
        pass
    
    def _load_las(self):
        pass
            
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
    
    def compose(self):
        
        pos = []
        color = []
        label = []
        
        for i in range(self.__len__()):
            p, c, l = self.get_raw(i)
            pos.append(p)
            color.append(c* (2**16))
            label.append((l>0)*(i+1))
        
        if len(pos) > 0:
            pos = np.vstack(pos)
            color = np.vstack(color)
            label = np.hstack(label)
            
            header = laspy.LasHeader(point_format=2, version="1.2")
            header.scales = np.array([0.01, 0.01, 0.01])
            header.add_extra_dim(laspy.ExtraBytesParams(name="id", type=np.float64))
            las = laspy.LasData(header)
            las.x = pos[:, 0]
            las.y = pos[:, 1]
            las.z = pos[:, 2]
            las.red = color[:, 0]
            las.green =  color[:, 1]
            las.blue = color[:, 2]
            las.id = label
            
            las.write("combined.las")

    
class Read_Las_from_Path(Las_Reader):
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
            

class Read_Las_from_Json(Las_Reader):
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
            
    


    
las_path = '../../notebooks/data/rocklas/prediction'
train_path = '../../notebooks/data/rocklas/raw'
json_path = '../../notebooks/data/rocklas/raw/train_split.json'