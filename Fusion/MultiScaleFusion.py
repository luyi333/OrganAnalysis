from turtle import forward
from load_data import ProbDataset
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import Parameter
import torch.nn as nn
import numpy as np
import os
from os.path import join


class MultiScaleFusion(nn.Module):
    """
    Fuse the Prob by Adaptive Weights
    """

    def __init__(self, class_num=15, extra_feature=1, hidden_size=32):
        super(MultiScaleFusion, self).__init__()

        self.in_features = class_num + extra_feature
        self.out_features = 1
        self.hidden_size = hidden_size
        
        self.mlp1 = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.out_features),
            # nn.ReLU()
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.out_features),
            # nn.ReLU()
        )

    def forward(self, images, code1, code2):

        w1 = self.mlp1(code1).unsqueeze(-1).unsqueeze(-1)
        w2 = self.mlp2(code2).unsqueeze(-1).unsqueeze(-1)

        image_fusion = w1*images[:, 0] + w2*images[:, 1]

        return image_fusion, w1, w2