import os
from os.path import join
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
import random
import pickle

def RandomCrop(img, patch_size=np.array([80, 160, 160])):
    img_shape = np.array(img.shape[2:])
    ranges = img_shape - patch_size
    start_pos = [random.randint(0, ranges[0]), random.randint(0, ranges[1]), random.randint(0, ranges[2])]
    end_pos = [start_pos[0]+patch_size[0], start_pos[1]+patch_size[1], start_pos[2]+patch_size[2]]

    img_crop = img[:,:, start_pos[0]:end_pos[0],
                        start_pos[1]:end_pos[1],
                        start_pos[2]:end_pos[2]]
    coord = [start_pos[0], end_pos[0], start_pos[1], end_pos[1], start_pos[2], end_pos[2]]
    return img_crop, coord

def SpecificCrop(img, coord):
    img_crop = img[:, coord[0]:coord[1], coord[2]:coord[3], coord[4]:coord[5]]
    return img_crop

def AutoPad(img, patch_size=np.array([80, 160, 160])):
    r = patch_size - np.array(img.shape[2:])
    need_to_pad = [max(0, r[0]), max(0, r[1]), max(0, r[2])]
    
    need_to_pad = [
        [need_to_pad[0] // 2, need_to_pad[0] - need_to_pad[0] // 2],
        [need_to_pad[1] // 2, need_to_pad[1] - need_to_pad[1] // 2],
        [need_to_pad[2] // 2, need_to_pad[2] - need_to_pad[2] // 2]
    ]

    return need_to_pad

def coding(dscore, class_num):
    code = np.zeros([class_num, class_num])
    for i in range(class_num):
        code[i, i] = 1
    dscore = np.expand_dims(np.concatenate(([0.], dscore), 0), 0)
    code =  np.concatenate((dscore, code), 0)
    code = code.transpose(1, 0)
    # print(code)
    return torch.tensor(code)

def data_prepare(data_root, trainers, fold):
    folder_list = [
        join(data_root[0], '%s__nnUNetPlansv2.1'%trainers[0]),
        join(data_root[1], '%s__nnUNetPlansv2.1'%trainers[1])
    ]

    train_list = [[], []]
    val_list = [[], []]

    for i in range(5):
        if i == fold:
            folder_1 = join(folder_list[0], 'fold_%d/validation_raw'%i)
            folder_2 = join(folder_list[1], 'fold_%d/up'%i)
            for filename in sorted(os.listdir(folder_1)):
                if filename[-3:] == 'npz':
                    # print(filename)
                    id = filename.split('.')[0][-3:]
                    # print(id)
                    val_list[0].append(join(folder_1, filename))
                    val_list[1].append(join(folder_2, 'MONETd2_%s.npz'%id))

        else:
            folder_1 = join(folder_list[0], 'fold_%d/validation_raw'%i)
            folder_2 = join(folder_list[1], 'fold_%d/up'%i)
            for filename in sorted(os.listdir(folder_1)):
                if filename[-3:] == 'npz':
                    id = filename.split('.')[0][-3:]
                    train_list[0].append(join(folder_1, filename))
                    train_list[1].append(join(folder_2, 'MONETd2_%s.npz'%id))


    dscore_files = [
        join(folder_list[0], 'detail_score.pkl'), 
        join(folder_list[1], 'detail_score.pkl')
    ]

    dscore_dicts = []

    for filename in dscore_files:
        with open(filename, 'rb') as fo:
            dict_data = pickle.load(fo, encoding='bytes')
        dscore_dicts.append(dict_data)

    return train_list, val_list, dscore_dicts

class ProbDataset_exp(Dataset):
    def __init__(self, file_list, target_folder, patch_size=np.array([80, 160, 160])):
        self.filelist = file_list
        self.targetfolder = target_folder
        self.patch_size = patch_size
    
    def __getitem__(self, index):
        data_dict = {}
        prob_file_1 = self.filelist[0][index]
        prob_file_2 = self.filelist[1][index]
        id = prob_file_1.split('.')[-2][-3:]
        target_file = join(self.targetfolder, 'MONET_%s.nii.gz'%id)
        
        data = np.expand_dims(np.load(prob_file_1, mmap_mode='r')['softmax'], 0)
        new = np.expand_dims(np.load(prob_file_2, mmap_mode='r')['softmax'], 0)
        data = np.concatenate((data, new), 0)
        
        target = sitk.GetArrayFromImage(sitk.ReadImage(target_file))
        target = np.expand_dims(target, 0).astype(np.int16)
        
        data_shape = np.array(target.shape[1:])

        if data_shape[0] >= self.patch_size[0] and data_shape[1] >= self.patch_size[1] and data_shape[1] >= self.patch_size[1]:

            data, coord = RandomCrop(data, patch_size=self.patch_size)
            target = SpecificCrop(target, coord)


        else:
            part_size = np.array([
                min(self.patch_size[0], data_shape[0]),
                min(self.patch_size[1], data_shape[1]),
                min(self.patch_size[2], data_shape[2]),
            ])
            data, coord = RandomCrop(data, patch_size=part_size)
            target = SpecificCrop(target, coord)


        need_to_pad = AutoPad(data, patch_size=np.array([80, 160, 160]))

        data = np.pad(data, ((0, 0),
                             (0, 0),
                             (need_to_pad[0][0], need_to_pad[0][1]),
                             (need_to_pad[1][0], need_to_pad[1][1]),
                             (need_to_pad[2][0], need_to_pad[2][1])),
                             'constant', constant_values=(0, 0))

        target = np.pad(target, ((0, 0),
                             (need_to_pad[0][0], need_to_pad[0][1]),
                             (need_to_pad[1][0], need_to_pad[1][1]),
                             (need_to_pad[2][0], need_to_pad[2][1])),
                             'constant', constant_values=(0, 0))
        
        data_dict['data'] = data.astype(np.float32)
        data_dict['target'] = target
        if need_to_pad == [[0,0],[0,0],[0,0]]:
            data_dict['padding'] = False
        else:
            data_dict['padding'] = True

        data_dict['id'] = id

        return data_dict

    def __len__(self):
        return len(self.filelist[0])


class ProbDataset_patch(Dataset):
    def __init__(self, folder_list, file_list, target_folder, patch_size=np.array([80, 160, 160])):
        self.filelist = file_list
        self.folderlist = folder_list
        self.targetfolder = target_folder
        self.patch_size = patch_size
    
    def __getitem__(self, index):
        data_dict = {}
        target_file = join(self.targetfolder, '%s.nii.gz'%(self.filelist[index][0:3]))

        for i, folder in enumerate(self.folderlist):
            new = np.expand_dims(np.load(join(folder, self.filelist[index]))['softmax'], 0)
            if i == 0:
                data = new
            else:
                data = np.concatenate((data, new), 0)
        
        target = sitk.GetArrayFromImage(sitk.ReadImage(target_file))
        target = np.expand_dims(target, 0).astype(np.int16)



        data_shape = np.array(target.shape[1:])
        if data_shape[0] >= self.patch_size[0] and data_shape[1] >= self.patch_size[1] and data_shape[1] >= self.patch_size[1]:

            data, coord = RandomCrop(data, patch_size=self.patch_size)
            target = SpecificCrop(target, coord)


        else:

            part_size = np.array([
                min(self.patch_size[0], data_shape[0]),
                min(self.patch_size[1], data_shape[1]),
                min(self.patch_size[2], data_shape[2]),
            ])
            data, coord = RandomCrop(data, patch_size=part_size)
            target = SpecificCrop(target, coord)


        need_to_pad = AutoPad(data, patch_size=np.array([80, 160, 160]))

        data = np.pad(data, ((0, 0),
                             (0, 0),
                             (need_to_pad[0][0], need_to_pad[0][1]),
                             (need_to_pad[1][0], need_to_pad[1][1]),
                             (need_to_pad[2][0], need_to_pad[2][1])),
                             'constant', constant_values=(0, 0))

        target = np.pad(target, ((0, 0),
                             (need_to_pad[0][0], need_to_pad[0][1]),
                             (need_to_pad[1][0], need_to_pad[1][1]),
                             (need_to_pad[2][0], need_to_pad[2][1])),
                             'constant', constant_values=(0, 0))
        
        data_dict['data'] = data.astype(np.float32)
        data_dict['target'] = target
        if need_to_pad == [[0,0],[0,0],[0,0]]:
            data_dict['padding'] = False
        else:
            data_dict['padding'] = True

        data_dict['id'] = self.filelist[index]

        return data_dict

    def __len__(self):
        return len(self.filelist)

class ProbDataset(Dataset):
    def __init__(self, folder_list, target_folder):
        self.filelist = sorted(os.listdir(folder_list[0]))
        self.folderlist = folder_list
        self.targetfolder = target_folder
    
    def __getitem__(self, index):
        data_dict = {}
        target_file = join(self.targetfolder, '%s.nii.gz'%(self.filelist[index][0:3]))

        for i, folder in enumerate(self.folderlist):
            new = np.expand_dims(np.load(join(folder, self.filelist[index]))['softmax'], 0)
            if i == 0:
                data = new
            else:
                data = np.concatenate((data, new), 0)
        
        target = sitk.GetArrayFromImage(sitk.ReadImage(target_file))

        data_dict['data'] = data.astype(np.float32)
        data_dict['target'] = np.expand_dims(target, 0).astype(np.int16)
        
        return data_dict

    def __len__(self):
        return len(self.filelist)