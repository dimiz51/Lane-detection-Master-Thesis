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
from torchvision.models import ViT_B_16_Weights

# Set seed for randomize functions (Ez reproduction of results)
random.seed(100)

# Map the 384 image size pretrained ViT to the 224 format for compatibility with the pretrained weights load function
def generate_map384():
        map_384 = {
                'encoder.ln.weight' : 'norm.weight',
                'encoder.ln.bias': 'norm.bias',
                'encoder.pos_embedding': 'pos_embedding',
                'conv_proj.weight': 'patch_embedding.proj.weight',
                'conv_proj.bias': 'patch_embedding.proj.bias', 
                }
        
        for i in range(12):
                prefix = f'encoder.layers.encoder_layer_{i}.'
                map_384[f'{prefix}ln_1.bias'] = f'transformer.layers.{i}.norm1.bias'
                map_384[f'{prefix}ln_1.weight'] = f'transformer.layers.{i}.norm1.weight'
                map_384[f'{prefix}ln_2.bias'] = f'transformer.layers.{i}.norm2.bias'
                map_384[f'{prefix}ln_2.weight'] =  f'transformer.layers.{i}.norm2.weight'
                map_384[f'{prefix}mlp.linear_1.bias'] = f'transformer.layers.{i}.linear1.bias'
                map_384[f'{prefix}mlp.linear_1.weight'] = f'transformer.layers.{i}.linear1.weight'
                map_384[f'{prefix}mlp.linear_2.bias'] = f'transformer.layers.{i}.linear2.bias'  
                map_384[f'{prefix}mlp.linear_2.weight'] = f'transformer.layers.{i}.linear2.weight'
                map_384[f'{prefix}self_attention.out_proj.bias'] = f'transformer.layers.{i}.self_attn.out_proj.bias'
                map_384[f'{prefix}self_attention.out_proj.weight'] =f'transformer.layers.{i}.self_attn.out_proj.weight'
                map_384[f'{prefix}self_attention.in_proj_bias'] = f'transformer.layers.{i}.self_attn.in_proj_bias'
                map_384[f'{prefix}self_attention.in_proj_weight'] =f'transformer.layers.{i}.self_attn.in_proj_weight'

        return map_384

# Generate the mapping dict renaming the pretrained weights layers names to the desired format
def generate_mapping_dict():
    mapping = {
        'norm.bias' : 'norm.bias',
        'norm.weight': 'norm.weight',
        'pos_embed': 'pos_embedding',
        'patch_embed.proj.bias': 'patch_embedding.proj.bias', 
        'patch_embed.proj.weight': 'patch_embedding.proj.weight',
    }

    for i in range(12):
        prefix = f'blocks.{i}.'

        mapping[f'{prefix}norm1.bias'] = f'transformer.layers.{i}.norm1.bias'
        mapping[f'{prefix}norm1.weight'] = f'transformer.layers.{i}.norm1.weight'
        mapping[f'{prefix}norm2.bias'] = f'transformer.layers.{i}.norm2.bias'
        mapping[f'{prefix}norm2.weight'] = f'transformer.layers.{i}.norm2.weight'
        mapping[f'{prefix}mlp.fc1.bias'] = f'transformer.layers.{i}.linear1.bias'
        mapping[f'{prefix}mlp.fc1.weight'] = f'transformer.layers.{i}.linear1.weight'
        mapping[f'{prefix}mlp.fc2.bias'] = f'transformer.layers.{i}.linear2.bias' 
        mapping[f'{prefix}mlp.fc2.weight'] = f'transformer.layers.{i}.linear2.weight'
        mapping[f'{prefix}attn.proj.bias'] = f'transformer.layers.{i}.self_attn.out_proj.bias'
        mapping[f'{prefix}attn.proj.weight'] = f'transformer.layers.{i}.self_attn.out_proj.weight'
        mapping[f'{prefix}attn.qkv.bias'] = f'transformer.layers.{i}.self_attn.in_proj_bias'
        mapping[f'{prefix}attn.qkv.weight'] = f'transformer.layers.{i}.self_attn.in_proj_weight'

    return mapping


# Resize the pretrained positional embeddings to desired dimensions
def resize_pretrained_pos(pretrained_dict, new_num_patches):
    
    pretrained_pos_embedding = pretrained_dict['pos_embedding']
    pretrained_pos_embedding = pretrained_pos_embedding[:, 1:, :]

    if pretrained_pos_embedding.shape[1] - 1 !=  new_num_patches:
        # Scale the positional embeddings by the ratio
        scaled_pos_embedding = F.interpolate(pretrained_pos_embedding.unsqueeze(0),
                                            size=(new_num_patches, pretrained_pos_embedding.shape[2]),
                                            mode='nearest').squeeze(0)

        # Create a new dictionary with the updated pos_embedding tensor
        pretrained_dict['pos_embedding'] = scaled_pos_embedding

        return pretrained_dict
    else:
        # Dispose the class token's pos embedding
        pretrained_pos_embedding_n = pretrained_pos_embedding[:, 1:, :]
        pretrained_dict['pos_embedding'] = pretrained_pos_embedding_n
        print('Resizing wasn\'t necessary for pre-trained positional embeddings.')
        return pretrained_dict

# Patch embedding class
class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, channels):
        super().__init__()

        self.image_size = image_size
        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError("image dimensions must be divisible by the patch size")
        self.grid_size = image_size[0] // patch_size, image_size[1] // patch_size
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, im):
        try:
            B, C, H, W = im.shape
        except:
            _, H, W = im.shape
        x = self.proj(im).flatten(2).transpose(1, 2)
        return x
    
    
    
# B-16 ViT Class
class ViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=1000, dim=768, depth=12, heads=12, 
                 mlp_dim=3072, dropout=0.1,load_pre = False, pre_trained_path = None):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        
        # Calculate the number of patches
        self.num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        # Define the patch embedding layer
        self.patch_embedding = PatchEmbedding((self.image_size,self.image_size),self.patch_size,self.dim, 1)
        
        
        # Define the positional embedding layer
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        self.pos_embedding = nn.init.trunc_normal_(self.pos_embedding,std= 0.02)
        
        # Define the transformer layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout),
            num_layers=depth
        )
        
        # Load model with pre-trained weights if flas is true else initiallize randomly
        if load_pre:
            self.load_pretrained_weights(pre_trained_path)
        else:
            # Randomly initialize patch embedding layer weights
            init.normal_(self.patch_embedding.proj.weight, std=0.02)

            # Initialize transformer layer weights
            for i in range(depth):
                layer = self.transformer.layers[i]
                # Multi-Head Attention weights
                init.normal_(layer.self_attn.in_proj_weight, std=0.02)
                init.normal_(layer.self_attn.out_proj.weight, std=0.02)
                # Feed-Forward layer weights
                init.normal_(layer.linear1.weight, std=0.02)
                init.normal_(layer.linear2.weight, std=0.02)
                # Layer Normalization weights
                init.constant_(layer.norm1.weight, 1)
                init.constant_(layer.norm2.weight, 1)

    def forward(self, x, return_features = True):
        # Apply the patch embedding layer
        
        x = self.patch_embedding(x)
        
        # Reshape the patches            
        x = x.flatten(2).transpose(1, 2)
                    
        # Dynamically expand pos embed across batch dimension
        if self.training:
            pos_embedding = nn.Parameter(self.pos_embedding.expand(x.shape[0], -1, -1))
            # Add the positional embeddings and use dropout
            x = (x.reshape(x.shape[0], -1, self.dim) + pos_embedding)
            x = self.dropout(x)
        else:
            # Batch forward for validation set
            pos_embedding = nn.Parameter(self.pos_embedding.expand(x.shape[0], -1, -1))
            # Add the positional embeddings and use dropout
            x = (x.reshape(x.shape[0], -1, self.dim) + pos_embedding)
            x = self.dropout(x)
                    
        # Apply the transformer layers
        x = self.transformer(x)
        
        # Apply layer normalization before returning the transformed features 
        x = self.norm(x)
        
        if return_features:
            return x
        
    # Load pre-trained weights method
    def load_pretrained_weights(self, pretrained_path: None):
        if pretrained_path:
            map_dict = generate_mapping_dict()
            pretrained = torch.load(pretrained_path)
        else:
            map_dict = generate_map384()
            pretrained = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1.get_state_dict(progress= True)
            print('Loading weights from ViT-B16_p224_fn384..!')
        model_state_dict = self.state_dict()
    
        # create new state dict with mapped keys
        new_state_dict = {}
        for key in pretrained:
            if key in map_dict:
                new_state_dict[map_dict[key]] = pretrained[key]
            else:
                if key in model_state_dict:
                    new_state_dict[key] = pretrained[key]

         # Test and see if resizing these is a good idea else keep the original randomly initialized weights
        new_state_dict = resize_pretrained_pos(new_state_dict, new_num_patches= self.num_patches) 
        
        # Load the mapped weights into our ViT model
        self.load_state_dict(new_state_dict, strict= True)
        print('Succesfully created ViT with pre-trained weights...!')
        
    # Freeze all layers except some
    def freeze_all_but_some(self, parameter_names):
        for name, param in self.named_parameters():
            if name not in parameter_names:
                param.requires_grad = False
                
    # Unfreeze some weights
    def unfreeze_some(self, parameter_names):
        for name, param in self.named_parameters():
            if name in parameter_names:
                param.requires_grad = True
    
    
    
    
                
            # Predict for one sample
            # pos_embedding = self.pos_embedding
            # # Add the positional embeddings and use dropout
            # x = (x.reshape(1, -1, 768) + pos_embedding)
            # x = self.dropout(x)