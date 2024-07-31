# 2022-1-22 16:49:37
#  

import os
import time
import torch
from shutil import copyfile
from copy import deepcopy

from AIG_MSPU.SSL.pixpro3d import PixPro, UNetEncoder, PixPro_Projector


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(params, epoch, model, optimizer, scheduler, sampler=None):
    state = {
        'opt': params,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }
    file_name = os.path.join(params['out_dir'], f'ckpt_epoch_{epoch}.pth')
    torch.save(state, file_name)
    copyfile(file_name, os.path.join(params['out_dir'], 'current.pth'))

def load_pretrained(model, pretrained_model):
    ckpt = torch.load(pretrained_model, map_location='cpu')
    state_dict = ckpt['model']
    model_dict = model.state_dict()

    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

def load_encoder_weights(unet_model, pretrained_model, params):
    pixpro_model = PixPro(UNetEncoder, params)
    load_pretrained(pixpro_model, pretrained_model)
    state_dict = pixpro_model.encoder.state_dict()
    new_state = deepcopy(unet_model.encoder.state_dict())
    for i in range(0, len(new_state.keys())):
        new_state[list(new_state.keys())[i]] = state_dict[list(state_dict.keys())[i]]
    unet_model.encoder.load_state_dict(new_state)
    del state_dict
    del pixpro_model
    
def load_encoder_weights_projector(mode, unet_model, pretrained_model, params):
    assert mode == 'pixpro_projector'
    pixpro_model = PixPro_Projector(UNetEncoder, params)
    load_pretrained(pixpro_model, pretrained_model)
    state_dict = pixpro_model.encoder.state_dict()
    new_state = deepcopy(unet_model.encoder.state_dict())
    for i in range(0, len(new_state.keys())):
        new_state[list(new_state.keys())[i]] = state_dict[list(state_dict.keys())[i]]
    unet_model.encoder.load_state_dict(new_state)
    del state_dict
    del pixpro_model

def load_encoder_weights_diffcin(mode, unet_model, pretrained_model, params):
    if mode == 'pixpro':
        pixpro_model = PixPro(UNetEncoder, params)
    else:
        assert False
    load_pretrained(pixpro_model, pretrained_model)
    state_dict = pixpro_model.encoder.state_dict()
    new_state = deepcopy(unet_model.encoder.state_dict())
    for i in range(0, len(new_state.keys())):
        if new_state[list(new_state.keys())[i]].shape==state_dict[list(state_dict.keys())[i]].shape:
            new_state[list(new_state.keys())[i]] = state_dict[list(state_dict.keys())[i]]
        else:
            new_state[list(new_state.keys())[i]][:, 0:1] = state_dict[list(state_dict.keys())[i]]
    
    unet_model.encoder.load_state_dict(new_state)
    del state_dict
    del pixpro_model

def WriteParams(filename, params):
    keys = list(params.keys())
    with open(filename, "w") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        f.write("\n-|- params in this experiment -|-\n")
        for key in keys:
            f.write(key+' : '+str(params[key])+'\n')

