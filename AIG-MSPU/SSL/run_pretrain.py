from AIG_MSPU.SSL.pixpro3d import PixPro, UNetEncoder
from AIG_MSPU.SSL.load_data import PreTrainDataset, RandomCrop
from AIG_MSPU.SSL.utils import AverageMeter, save_checkpoint

import numpy as np
import time
import tableprint as tp
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler, MultiStepLR, CosineAnnealingLR

params = {
    # pixpro
    'pixpro_p': 1.,
    'pixpro-momentum': 0.99,
    'pixpro-pos-ratio': 0.7,
    'pixpro-clamp-value': 0,
    'pixpro_transform_layer': 0,
    'pixpro-ins-loss-weight': 0,
    # train
    'device': 'cuda:0',
    'step_size': 100,              
    'epochs': 100,
    'start_epoch': 1,
    'batch_size': 1,
    'base_learning_rate': 0.001,
    'momentum': 0.9,               
    'weight_decay': 1e-4,
    'warmup_epoch':0,
    'save_freq': 10,
    # data
    'dataset_folder': '/home4/usr/datasets/available/', 
    'data_shape': np.array([120, 512, 512]),
    'output_dir': '/SSL'
}

def train(params):

    # Data Setting
    dataset = PreTrainDataset(params['dataset_folder'], params['data_shape'])
    dataloader = DataLoader(dataset=dataset, num_workers=2, pin_memory=False, batch_size=params['batch_size'], shuffle=True)

    # Network Setting
    model = PixPro(UNetEncoder, params).to(params['device'])

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=params['batch_size'] / 1 * params['base_learning_rate'],
                                momentum=params['momentum'],
                                weight_decay=params['weight_decay'],)
    
    # Scheduler
    scheduler = CosineAnnealingLR(optimizer=optimizer,
                                  eta_min=0.000001,
                                  T_max=(params['epochs'] - params['warmup_epoch']) * params['step_size'])

    # Train
    for epoch in range(params['start_epoch'], params['epochs'] +1):
            tp.banner('EPOCH %d'%epoch)
            train_epoch(epoch, dataloader, model, optimizer, scheduler, params)
            if (epoch % params['save_freq'] == 0 or epoch == params['epochs']):
                print('Saving...')
                save_checkpoint(params, epoch, model, optimizer, scheduler)


def train_epoch(epoch, dataloader, model, optimizer, scheduler, params):
    model.train()
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    end = time.time()
    epoch_loss = 0.0
    num = 0
    for i, img in enumerate(tqdm(dataloader)):
        img_1, coord_1 = RandomCrop(img)
        img_2, coord_2 = RandomCrop(img)
        img_1 = img_1.to(params['device']).to(torch.float32)
        coord_1 = coord_1.to(params['device']).to(torch.float32)
        img_2 = img_2.to(params['device']).to(torch.float32)
        coord_2 = coord_2.to(params['device']).to(torch.float32)
        
        loss = model(img_1, img_2, coord_1, coord_2)
        if loss.detach().item() != 0.0:
            epoch_loss += loss.detach().item()
            num += 1

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        scheduler.step()

        loss_meter.update(loss.item(), img_1.size(0))
        batch_time.update(time.time() - end)
    
    print('Average Loss:', epoch_loss / num)


                      
if __name__ == '__main__':
    train(params)