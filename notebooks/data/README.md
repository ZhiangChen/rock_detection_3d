# create your own dataset

1. create a folder of your dataset under data directory and two sub-folders  
```
- rocklas
	- processed
	- raw
```

2. copy all your .las annotations under raw directory  

3. create a json file to include your training, validation, and test splits under raw  
```
import json
import os
from random import shuffle
las_files = [f for f in os.listdir('.') if f.endswith('.las')]
shuffle(las_files)
nm = len(las_files)
train_files = las_files[:int(nm*0.6)]
valid_files = las_files[int(nm*0.6):int(nm*0.8)]
test_files = las_files[int(nm*0.8):]
with open('train_split.json', 'w') as f:
    json.dump(train_files, f)

with open('valid_split.json', 'w') as f:
    json.dump(valid_files, f)

with open('test_split.json', 'w') as f:
    json.dump(test_files, f)

```

4. create a dataloading python script under rock_detection_3d/datasets/segmentation. E.g. rock_las.py   


5. create a notebook to check your dataloading script under rock_detection_3d/notebooks. E.g. dataset_test.ipynb  


References:  
- pytorch-geometric: create your own datasets: https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html
- torch-points3d: create a new dataset: https://torch-points3d.readthedocs.io/en/latest/src/tutorials.html#create-a-new-dataset
- example shapenet.py: https://github.com/nicolas-chaulet/torch-points3d/blob/master/torch_points3d/datasets/segmentation/shapenet.py

If you change your code, you need to manually remove all data files to create new data files. 
