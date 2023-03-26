# Imports 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import cv2
import os
import json
import math
import torch.nn.init as init
from einops import rearrange


# Set seed for randomize functions (Ez reproduction of results)
random.seed(100)

# Import TuSimple loader
import sys
sys.path.insert(0,'../resources/')
from tusimple import TuSimple


# MLP decoder for the ViT encoder (ViT+Linear Reg Segmenter)
class DecoderMLP(nn.Module):
    def __init__(self, n_classes, d_encoder, patch_size = 16, image_size = (640,640)):
        super().__init__()
        self.image_size = image_size
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_cls = n_classes
        self.num_patches = (image_size[0] // patch_size) ** 2

        self.mlp = nn.Sequential(
                    nn.Linear(384, 512),
                    nn.BatchNorm1d(self.num_patches),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(self.num_patches),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                    nn.Linear(256, n_classes)
                    )
        
        self.apply(self.weights_init)

    def forward(self, x):
        H, W = self.image_size
        GS = H // self.patch_size
        x = self.mlp(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=GS)

        return x
    
    def weights_init(self,m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            init.normal_(m.weight, 1.0, 0.02)
            init.constant_(m.bias, 0)