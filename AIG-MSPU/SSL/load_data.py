# 2022-1-21 16:02:32
#  
# Data Loading Part For PixPro Pretraining 

import os
from os.path import join
import numpy as np
import torch.utils.data as data
import SimpleITK as sitk
import pandas as pd

from scipy.ndimage import zoom
import random
import torch

class PreTrainDataset(data.Dataset):
    """
    Dataset for 3d volume
    """
    def __init__(self, data_folder, data_shape):
        self.data_folder = data_folder
        self.file_list = sorted(os.listdir(data_folder))
        self.len = len(self.file_list)
        self.data_shape = data_shape
    def __getitem__(self, index):
        img = sitk.ReadImage(join(self.data_folder, self.file_list[index]))
        array = sitk.GetArrayFromImage(img)
        factor = self.data_shape / array.shape
        array = zoom(array, factor, order=1)
        array = np.expand_dims(array, 0)
        return array
    def __len__(self):
        return self.len

class PreTrainDataset_list(data.Dataset):
    """
    Dataset for 3d volume | train list from txt file
    """
    def __init__(self, train_list_file, data_shape, patch_size):
        with open(train_list_file, 'r') as f:
            a = f.readlines()
            a = [x[0:-1] for x in a]
        self.file_list = a
        self.len = len(self.file_list)
        self.data_shape = data_shape
        self.patch_size = patch_size

    def __getitem__(self, index):
        img = sitk.ReadImage(self.file_list[index])
        array = sitk.GetArrayFromImage(img)

        if array.shape[0] < self.patch_size[0] or array.shape[1] < self.patch_size[1] or array.shape[2] < self.patch_size[2]:
            factor = self.data_shape / array.shape
            array = zoom(array, factor, order=1)
        
        array = np.expand_dims(array, 0)
        return array
    def __len__(self):
        return self.len

class PreTrainDataset_list_batch(data.Dataset):
    """
    Dataset for 3d volume | train list from txt file
    """
    def __init__(self, train_list_file, data_shape, patch_size):
        with open(train_list_file, 'r') as f:
            a = f.readlines()
            a = [x[0:-1] for x in a]
        self.file_list = a
        self.len = len(self.file_list)
        self.data_shape = data_shape
        self.patch_size = patch_size

    def __getitem__(self, index):
        img = sitk.ReadImage(self.file_list[index])
        array = sitk.GetArrayFromImage(img)

        if array.shape[0] < self.data_shape[0] or array.shape[1] < self.data_shape[1] or array.shape[2] < self.data_shape[2]:
            factor = self.data_shape / array.shape
            array = zoom(array, factor, order=1)
        else:
            array = RandomCrop_in_dataset(array, patch_size=self.data_shape)
        array = np.expand_dims(array, 0).astype(np.float32)
        return array
    def __len__(self):
        return self.len

def RandomCrop_in_dataset(img, patch_size=np.array([80, 160, 160])):
    img_shape = np.array(img.shape)
    ranges = img_shape - patch_size
    start_pos = [random.randint(0, ranges[0]), random.randint(0, ranges[1]), random.randint(0, ranges[2])]
    end_pos = [start_pos[0]+patch_size[0], start_pos[1]+patch_size[1], start_pos[2]+patch_size[2]]
    img_crop = img[start_pos[0]:end_pos[0],
                        start_pos[1]:end_pos[1],
                        start_pos[2]:end_pos[2]]
    return img_crop

def RandomCrop(img, patch_size=np.array([80, 160, 160])):
    img_shape = np.array(img.shape[2:])
    ranges = img_shape - patch_size
    start_pos = [random.randint(0, ranges[0]), random.randint(0, ranges[1]), random.randint(0, ranges[2])]
    end_pos = [start_pos[0]+patch_size[0], start_pos[1]+patch_size[1], start_pos[2]+patch_size[2]]
    img_crop = img[:,:, start_pos[0]:end_pos[0],
                        start_pos[1]:end_pos[1],
                        start_pos[2]:end_pos[2]]
    coord = torch.tensor([[start_pos[0], start_pos[1], start_pos[2], end_pos[0], end_pos[1], end_pos[2]]])
    # print(img_crop.shape, coord.shape)
    return img_crop, coord
