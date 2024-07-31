# An example to generate the volume fraction of base materials from a SECT volume.
import json
import os
from os.path import join

import numpy as np
import SimpleITK as sitk
import torch
from decompose import DecomposeTo3Materials
from network.DECT_CNN import UNet

if __name__ == '__main__':

    img_file = '/nvme1date/usr/MONET//TCGA-D1-A163_0000.nii.gz'
    range_file = '/nvme1date/usr/MONET//TCGA-D1-A163.txt'
    coefficient_file = '/nvme1date/usr/MONET/Corrections/TCGA-D1-A163.json'
    save_name = '/nvme1date/usr/MONET/ResultsMaterials/TCGA-UCEC/TCGA-D1-A163.npy'
    
    print('img_file:', img_file)
    print('range_file:', range_file)
    print('coefficient_file:', coefficient_file)
    print('save_name:', save_name)
    
    checkpoint_dir = 'network\SECT_DECT.pth'
    net = UNet(input_channel=1, output_channel=2).cuda()
    net.load_state_dict(torch.load(checkpoint_dir))
    
    with open(coefficient_file, 'r+') as f:
        coef_dict = json.loads(f.read())
        
    linear_coefficient = {'Air':     (coef_dict['Air'][0], coef_dict['Air'][1]),
                         'Adipose': (coef_dict['Adipose'][0], coef_dict['Adipose'][1]),
                         'Muscle':  (coef_dict['Muscle'][0], coef_dict['Muscle'][1]),
                         'Bone':    (coef_dict['Bone'][0], coef_dict['Bone'][1])}
    
    print(linear_coefficient)
    
    z_range = np.loadtxt(range_file).astype(np.int32)
    result = DecomposeTo3Materials(img_file, linear_coefficient, net, z_range)
    np.save(save_name, result)
