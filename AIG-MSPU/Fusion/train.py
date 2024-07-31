from cv2 import split
from tqdm import tqdm
from load_data import ProbDataset, ProbDataset_patch, coding, ProbDataset_exp, data_prepare
import SimpleITK as sitk
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import Parameter
import numpy as np
import os
from os.path import join
import tableprint as tp
import datetime
import pandas as pd
import pickle
import shutil

from MultiScaleFusion import MultiScaleFusion

# Data Setting
fold = 0
result_dir = 'results'
if os.path.isdir(result_dir):
    pass
else:
    os.makedirs(result_dir)
    os.makedirs(join(result_dir, 'ckpt'))
    os.makedirs(join(result_dir, 'prediction'))

data_root = [
    '/nvme1date/usr/nnunet_data/result/nnUNet/3d_fullres/Task517_MONET/',
    '/nvme1date/usr/nnunet_data/result/nnUNet/3d_fullres/Task520_MONETd2/'
]

target_folder = "/nvme1date/usr/nnunet_data/result/nnUNet/3d_fullres/Task517_MONET/Trainer_AIG_MSPU/gt/"

trainers = [
    'Trainer_AIG_MSPU',
    'Trainer_AIG_MSPU'
]

train_list, val_list, dscore_dicts = data_prepare(data_root, trainers, fold)

# Train Setting
max_epoch = 1000
lr = 1e-3
device = 'cuda:0'


# Data Loading
dataset_tr = ProbDataset_exp(train_list, target_folder, patch_size=np.array([80, 160, 160]))
dataloader_tr = DataLoader(dataset=dataset_tr, batch_size=1, num_workers=8)

dataset_val = ProbDataset_exp(val_list, target_folder, patch_size=np.array([80, 160, 160]))
dataloader_val = DataLoader(dataset=dataset_val, batch_size=1, num_workers=8)

# Network
network = MultiScaleFusion(class_num=15, extra_feature=1, hidden_size=32).to(device)
print(network)

# Loss
loss = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})

# Optimizer
optimizer = torch.optim.Adam(network.parameters(), lr=lr)

# Log
shutil.copy(
    '/home/usr/MONET/FusionNew/train_exp.py',
    result_dir
)
log_file = join(result_dir, 'log.txt')
with open(log_file, "w") as f:
        f.write('Fusion Experiment')

best_val_loss = 100.
for epoch in range(max_epoch):
    network.train()
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n', time)
    tp.banner('Epoch: %d          '%epoch)
    with open(log_file, "a") as f:
        f.write("\n\n%s\n--epoch : %d--"%(time, epoch))
    
    loss_list = []
   
    # Train
    print('Training...')
    for data_dict in tqdm(dataloader_tr):
        data = data_dict['data'].to(device)
        target = data_dict['target'].to(device)
        padding = data_dict['padding'][0].item()
        id = data_dict['id'][0]
        dscore_1 = dscore_dicts[0]['MONET_'+id]
        dscore_2 = dscore_dicts[1]['MONETd2_'+id]

        code_1 = coding(dscore_1, 15).unsqueeze(0).to(device).to(torch.float32)
        code_2 = coding(dscore_2, 15).unsqueeze(0).to(device).to(torch.float32)

        predict, w1, w2 = network(data, code_1, code_2)
        predict = torch.softmax(predict, 1)

        l = loss(predict, target)

        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        loss_list.append(l.item())

    print('Train Done!')
    mean_train_loss = np.mean(loss_list)
    print('Mean Train Loss:', mean_train_loss)
    print('w1', w1.flatten())
    print('w2', w2.flatten())

    with open(log_file, "a") as f:
        f.write("\ntrain_loss:%05f"%mean_train_loss)
        f.write('\nw1:%s'%str(w1.flatten()))
        f.write('\nw2:%s'%str(w2.flatten()))

    state = {
        'net': network.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(state, join(result_dir, 'ckpt', '%d.pth'%epoch))

    # Validation
    if (epoch + 1) % 5 == 0:
        run_val = True
    else:
        run_val = False
    
    if run_val:
        network.eval()
        loss_val_list = []
        loss_ref_list = []
        print('Validating...')
        for data_dict in tqdm(dataloader_val):

            data = data_dict['data'].to(device)
            target = data_dict['target'].to(device)
            padding = data_dict['padding'][0].item()
            id = data_dict['id'][0]
            dscore_1 = dscore_dicts[0]['MONET_'+id]
            dscore_2 = dscore_dicts[1]['MONETd2_'+id]

            code_1 = coding(dscore_1, 15).unsqueeze(0).to(device).to(torch.float32)
            code_2 = coding(dscore_2, 15).unsqueeze(0).to(device).to(torch.float32)
            
            predict_val, w1, w2 = network(data, code_1, code_2)
            predict_val = torch.softmax(predict_val, 1)

            loss_val = loss(predict_val, target)
            loss_ref = loss(data[:, 0, :, :, :, :], target)

            loss_val_list.append(loss_val.item())
            loss_ref_list.append(loss_ref.item())

            # break

        print('Validation Done!')
        mean_val_loss = np.mean(loss_val_list)
        mean_ref_loss = np.mean(loss_ref_list)
        print('Mean Val Loss:', mean_val_loss)
        print('Mean Ref Loss:', mean_ref_loss)

        with open(log_file, "a") as f:
            f.write("\nval_loss:%05f"%mean_val_loss)
            f.write('\nw1:%s'%str(w1.flatten()))
            f.write('\nw2:%s'%str(w2.flatten()))

        if mean_val_loss < best_val_loss:
            print('This is best!')
            best_val_loss = mean_val_loss
            state = {
                'net': network.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(state, join(result_dir, 'best.pth'))

