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

# Masks transformer class
class MaskTransformer(nn.Module):
    def __init__(self, image_size = (640,640) ,n_classes = 2, patch_size = 16, depth = 6 ,heads = 8, dim_enc = 768, dim_dec = 768, mlp_dim = 3072, dropout = 0.1):
        super(MaskTransformer, self).__init__()
        self.dim = dim_enc
        self.patch_size = patch_size
        self.depth = depth
        self.class_n = n_classes
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.d_model = dim_dec
        self.scale = self.d_model ** -0.5
        self.att_heads = heads
        self.image_size = image_size
        
        # Define the transformer blocks
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(dim_dec, heads, mlp_dim, dropout)
            for _ in range(self.depth)
            ])
        
        # Learnable Class embedding parameter
        self.cls_emb = nn.Parameter(torch.randn(1, n_classes,dim_dec))
        
        # Projection layers for patch embeddings and class embeddings
        self.proj_dec = nn.Linear(dim_enc,dim_dec)
        self.proj_patch = nn.Parameter(self.scale * torch.randn(dim_dec, dim_dec))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(dim_dec, dim_dec))
        
        # Normalization layers
        self.decoder_norm = nn.LayerNorm(dim_dec)
        self.mask_norm = nn.LayerNorm(n_classes)
        
        
        # Initialize weights from a random normal distribution for all layers and the class embedding parameter
        self.apply(self.init_weights)
        init.normal_(self.cls_emb, std=0.02)
        
    # Init weights method
    @staticmethod
    def init_weights(module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_in')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
            
    def forward(self, x):
        H, W = self.image_size
        GS = H // self.patch_size

        # Project embeddings to mask transformer dim size and expand class embedding(by adding the batch dim) to match these 
        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        
        # Add the learnable class embedding to the patch embeddings and pass through the transformer blocks
        x = torch.cat((x, cls_emb), 1)
        for blk in self.transformer_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # Split output tensor into patch embeddings and the transformer patch level class embeddings
        patches, cls_seg_feat = x[:, : -self.class_n], x[:, -self.class_n :]
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes

        # Perform L2 Normalizations over the two tensors
        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

        # 1. Calculate patch level class scores(as per dot product) by between the normalized patch tensors and the normalized class embeddings
        # 2. Reshape the output from (batch,number of patches, classes) to (batch size, classes, height, width)
        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks)
        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))

        return masks       