import os
import os.path as osp
import shutil
import json
from tqdm.auto import tqdm as tq
from itertools import repeat, product
import numpy as np
import torch
import laspy

from torch_geometric.data import Data, InMemoryDataset, extract_zip
from torch_geometric.io import read_txt_array
import torch_geometric.transforms as T
from torch_points3d.core.data_transform import SaveOriginalPosId
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker
from torch_points3d.datasets.base_dataset import BaseDataset, save_used_properties
from torch_points3d.utils.download import download_url


class RockLAS(InMemoryDataset):
    r"""
    Args:
        root (string): Root directory where the dataset should be saved.
        categories (string or [string], optional): The category of the
            models (obj:`"PBR"`)
        include_color (bool, optional): If set to :obj:`False`, will not
            include color vectors as input features. (default: :obj:`True`)
        split (string, optional): If :obj:`"train"`, loads the training
            dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"trainval"`, loads the training and validation dataset.
            If :obj:`"test"`, loads the test dataset.
            (default: :obj:`"trainval"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """


    category_ids = {
        "PBR": "02691156",
    }

    seg_classes = {
        "PBR": [0, 1],  # 0: isPBR; 1: isPedestal
    }

    def __init__(
        self,
        root,
        include_color=True,
        split="trainval",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        is_test=False,
    ):
        categories = list(self.category_ids.keys())
        self.categories = categories
        self.is_test = is_test
        super(RockLAS, self).__init__(
            root, transform, pre_transform, pre_filter)

        if split == "train":
            path = self.processed_paths[0]
            raw_path = self.processed_raw_paths[0]
        elif split == "val":
            path = self.processed_paths[1]
            raw_path = self.processed_raw_paths[1]
        elif split == "test":
            path = self.processed_paths[2]
            raw_path = self.processed_raw_paths[2]
        elif split == "trainval":
            path = self.processed_paths[3]
            raw_path = self.processed_raw_paths[3]
        else:
            raise ValueError(
                (f"Split {split} found, but expected either " "train, val, trainval or test"))
        
        
        self.data, self.slices, self.y_mask = self.load_data(
            path, include_color)

        # We have perform a slighly optimzation on memory space of no pre-transform was used.
        # c.f self._process_filenames
        
        if os.path.exists(raw_path):
            self.raw_data, self.raw_slices, _ = self.load_data(
                raw_path, include_color)
        else:
            self.get_raw_data = self.get

        
    def load_data(self, path, include_color):
        '''This function is used twice to load data for both raw and pre_transformed
        '''
        data, slices = torch.load(path)
        data.x = data.x if include_color else None

        y_mask = torch.zeros(
            (len(self.seg_classes.keys()), 50), dtype=torch.bool)
        for i, labels in enumerate(self.seg_classes.values()):
            y_mask[i, labels] = 1

        return data, slices, y_mask
    
    def get_raw_data(self, idx, **kwargs):  # this may be not needed
        data = self.raw_data.__class__()

        if hasattr(self.raw_data, '__num_nodes__'):
            data.num_nodes = self.raw_data.__num_nodes__[idx]

        for key in self.raw_data.keys:
            item, slices = self.raw_data[key], self.raw_slices[key]
            start, end = slices[idx].item(), slices[idx + 1].item()
            # print(slices[idx], slices[idx + 1])
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.raw_data.__cat_dim__(key, item)] = slice(start, end)
            elif start + 1 == end:
                s = slices[start]
            else:
                s = slice(start, end)
            data[key] = item[s]
        return data
    
    @property
    def processed_file_names(self):
        # this should be created instead of being listed
        # these processed files are pytorch data files after sampling and transforms
        return ['processed_train.pt', 'processed_valid.pt', 'processed_test.pt', 'processed_trainvalid.pt']
        
    @property
    def raw_file_names(self):
        # this should be created instead of being listed
        # those raw files are pytorch data files before sampling or transforms
        return ['raw_train.pt', 'raw_valid.pt', 'raw_test.pt', 'raw_trainvalid.pt']  
        
    @property
    def processed_raw_paths(self):
        # this should be created instead of being listed
        # this gives the paths of processed raw data files 
        return [os.path.join(self.processed_dir, f) for f in self.raw_file_names]
    
    def process(self):
        if self.is_test:
            return
        raw_trainval = []
        trainval = []
        for i, split in enumerate(["train", "valid", "test"]):
            json_path = osp.join(self.raw_dir, f"{split}_split.json")
            with open(json_path, "r") as f:
                filenames = json.load(f)
            data_raw_list, data_list = self._process_filenames(sorted(filenames))
            
            if split == "train" or split == "valid":
                if len(data_raw_list) > 0:
                    raw_trainval.append(data_raw_list)
                trainval.append(data_list)
            
            self._save_data_list(data_list, self.processed_paths[i])
            self._save_data_list(data_raw_list, self.processed_raw_paths[i], save_bool=len(data_raw_list) > 0)

        self._save_data_list(self._re_index_trainval(trainval), self.processed_paths[3])
        self._save_data_list(self._re_index_trainval(
            raw_trainval), self.processed_raw_paths[3], save_bool=len(raw_trainval) > 0)
    
    def _process_filenames(self, filenames):
        data_raw_list = []
        data_list = []
        
        has_pre_transform = self.pre_transform is not None
        
        for id_scan, name in enumerate(filenames):
            # print(name)
            las_file = os.path.join(self.raw_dir, name)
            las = laspy.read(las_file)
            pos = np.array((las.x.scaled_array(), las.y.scaled_array(), las.z.scaled_array())).transpose().astype(np.float64)
            x = np.array((las.red, las.green, las.blue)).transpose().astype(np.float64) / (2**16)
            try:
                y = (las.isPBR==0)*1
            except AttributeError:
                y = np.logical_not((las.notPBR==1))*1
            pos = torch.from_numpy(pos) 
            y = torch.from_numpy(y)
            x = torch.from_numpy(x) 
                
            category = torch.ones(x.shape[0], dtype=torch.long) * 0
            id_scan_tensor = torch.from_numpy(np.asarray([id_scan])).clone()
            
            # normalize scale
            center = pos.mean(axis=0)
            pos = pos - center
            scale = ((1 / pos.abs().max()) * 0.999999).reshape((1))
            pos = pos * scale
            normalize_attr = {'center': center, 'scale': scale}
            
            data = Data(pos=pos, x=x, y=y, category=category, id_scan=id_scan_tensor, scale=scale, center=center, file_name=name)
            data = SaveOriginalPosId()(data)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            data_raw_list.append(data.clone() if has_pre_transform else data)
            if has_pre_transform:
                data = self.pre_transform(data)
                data.x = data.x.float()  # cast float64 to float32 after  transformation; if before, there is a server data precision loss issue
                data.pos = data.pos.float()
                data_list.append(data)
        if not has_pre_transform:
            return [], data_raw_list
        return data_raw_list, data_list
    
    def _save_data_list(self, datas, path_to_datas, save_bool=True):
        if save_bool:
            torch.save(self.collate(datas), path_to_datas)
        
    def _re_index_trainval(self, trainval):
        if len(trainval) == 0:
            return trainval
        train, val = trainval
        for v in val:
            v.id_scan += len(train)
        assert (train[-1].id_scan + 1 ==
                val[0].id_scan).item(), (train[-1].id_scan, val[0].id_scan)
        return train + val

    def __repr__(self):
        return "{}({}, categories={})".format(self.__class__.__name__, len(self), self.categories)


class RockLASDataset(BaseDataset):
    """ Wrapper around RockLAS that creates train and test datasets.

    Parameters
    ----------
    dataset_opt: omegaconf.DictConfig
        Config dictionary that should contain

            - dataroot
            - normal: bool, include normals or not
            - pre_transforms
            - train_transforms
            - test_transforms
            - val_transforms
    """

    # FORWARD_CLASS = "forward.shapenet.ForwardShapenetDataset" # unknow usage of this

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)
        is_test = dataset_opt.get("is_test", False)

        self.train_dataset = RockLAS(
            self._data_path,
            include_color=dataset_opt.color,
            split="train",
            pre_transform=self.pre_transform,
            transform=self.train_transform,
            is_test=is_test,
        )
        
        
        self.val_dataset = RockLAS(
            self._data_path,
            include_color=dataset_opt.color,
            split="val",
            pre_transform=self.pre_transform,
            transform=self.test_transform,  # valid dataset has the same transform as the test dataset
            is_test=is_test,
        )

        self.test_dataset = RockLAS(
            self._data_path,
            include_color=dataset_opt.color,
            split="test",
            transform=self.test_transform,
            pre_transform=self.pre_transform,
            is_test=is_test,
        )
        
        
        self._categories = self.train_dataset.categories

    @property  # type: ignore
    @save_used_properties
    def class_to_segments(self):
        classes_to_segment = {}
        for key in self._categories:
            classes_to_segment[key] = RockLAS.seg_classes[key]
        return classes_to_segment

    @property
    def is_hierarchical(self):
        return len(self._categories) > 1

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        return SegmentationTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)
