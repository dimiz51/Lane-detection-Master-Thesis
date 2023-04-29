# APPROACH LIKE : https://github.com/rstrudel/segmenter (vit base 16 + mlp decoder with/without pretrained weights for vit)


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
from torch.utils.data import DataLoader
import matplotlib as plt

from torch.utils.data import DataLoader
from torchmetrics import F1Score,JaccardIndex
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# Import TuSimple loader
import sys
sys.path.insert(0,'../resources/')
from tusimple import TuSimple
from mlp_decoder import DecoderMLP
import utils

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
                'class_token': 'cls_token'
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

    if pretrained_pos_embedding.shape[1] - 1!=  new_num_patches:
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
    
    
# ViT Class
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
        self.patch_embedding = PatchEmbedding((self.image_size,self.image_size),self.patch_size,self.dim, 3)
        
        # Define the positional embedding layer
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        
        # Define the class embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        
        # Define the transformer layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout),
            num_layers=depth
        )
        
        # Load model with pre-trained weights if flas is true else initiallize randomly
        if load_pre:
            self.load_pretrained_weights(pre_trained_path)

    def forward(self, x, return_features = True):
        # Apply the patch embedding layer
        
        x = self.patch_embedding(x)
        
        # Reshape the patches            
        x = x.flatten(2).transpose(1, 2)
                    
        # Dynamically expand pos embed across batch dimension
        if self.training:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            pos_embedding = nn.Parameter(self.pos_embedding.expand(x.shape[0], -1, -1))
            x = torch.cat((cls_tokens, x.permute(0,2,1)), dim=1)
            # Add the positional embeddings and use dropout
            x = (x + pos_embedding)
            x = self.dropout(x)
        else:
            # Add learnable class token per patch
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x.permute(0,2,1)), dim=1)
            
            pos_embedding = nn.Parameter(self.pos_embedding.expand(x.shape[0], -1, -1))

            # Add the positional embeddings and use dropout
            x = (x + pos_embedding)
            
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
        new_state_dict = resize_pretrained_pos(new_state_dict, new_num_patches= self.num_patches + 1) 
        
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
    
    # Count trainable parameters
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
# Plot metrics function 
def plot_metrics(train_losses, val_losses, train_f1, val_f1, train_iou, val_iou, save_path = '../plots/'):
    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation')
    plt.xlabel('Epochs (bins of 5)')
    plt.ylabel('Loss')
    plt.xticks(range(0, len(train_losses) + 1, 5))
    plt.legend()
    
    if save_path is not None:
        filename = 'loss_plot.png'
        plt.savefig(os.path.join(save_path, filename))
        
    # plt.show()

    # Plot training and validation F1 scores
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_f1) + 1), train_f1, label='Train')
    plt.plot(range(1, len(val_f1) + 1), val_f1, label='Validation')
    plt.xlabel('Epochs (bins of 5)')
    plt.ylabel('F1 Score')
    plt.xticks(range(0, len(train_f1) + 1, 5))
    plt.legend()
    
    if save_path is not None:
        filename = 'f1_plot.png'
        plt.savefig(os.path.join(save_path, filename))
        
    # plt.show()

    # Plot training and validation IoU scores
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_iou) + 1), train_iou, label='Train')
    plt.plot(range(1, len(val_iou) + 1), val_iou, label='Validation')
    plt.xlabel('Epochs (bins of 5)')
    plt.ylabel('IoU Score')
    plt.xticks(range(0, len(train_iou) + 1, 5))
    plt.legend()
    
        
    if save_path is not None:
        filename = 'iou_plot.png'
        plt.savefig(os.path.join(save_path, filename))
        
    # plt.show()
    
class Segmenter(nn.Module):
    def __init__(self,vit_variant, mlp_head, image_size = (448,448)):
        super().__init__()
        self.transformer = vit_variant
        self.mlp = mlp_head
        self.image_size = image_size
        self.lane_threshold = 0.5
        self.activation = nn.Sigmoid()
        
    # Forward pass of the pipeline
    def forward(self, im):
        H, W = self.image_size
                
        # Transform standardized feature maps using the ViT
        x = self.transformer(im)
        
        # Remove the learnable class token before patch classification
        x = x[:, 1:]
        
        # Perform patch level classification (0 for background/ 1 for lane)
        x = self.mlp(x)

        # Interpolate patch level class annotatations to pixel level and transform to original image size
        logits = F.interpolate(x, size=(H, W), mode="bilinear")
        
        probs = self.activation(logits)

        return logits, probs
        
        
    # Count pipeline trainable parameters
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    # Load trained model
    def load_weights(self,path): 
        self.load_state_dict(torch.load(path,map_location=torch.device('cpu')))

# Custom training function for the pipeline with schedule and augmentations
def train(model, train_loader, val_loader = None, num_epochs=10, lr=0.01, weight_decay=0, SGD_momentum = 0.9, lr_scheduler=False, lane_weight = None, save_path = None):
    # Set up loss function and optimizer
    criterion =  nn.BCEWithLogitsLoss(pos_weight= lane_weight)

    #create seperate parameter groups for the different networks
    mlp_params = [p for p in model.mlp.parameters() if p.requires_grad]
    vit_params = [p for p in model.transformer.parameters() if p.requires_grad]

    #define the optimizer with different learning rates for the parameter groups
    optimizer = optim.SGD([
        {'params' : mlp_params, 'lr' : lr},
        {'params' : vit_params, 'lr' : lr}
    ], momentum=SGD_momentum, weight_decay= weight_decay)
    
    # Define your learning rate scheduler
    if lr_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True, threshold=1e-4, min_lr=[0.001, 0.0005])

    # Set up device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    f1_score = F1Score(task="binary").to(device)
    iou_score = JaccardIndex(task= 'binary').to(device)
    
    gt_augmentations = transforms.Compose([transforms.RandomRotation(degrees=(10, 30)),
                                              transforms.RandomHorizontalFlip()])
    
    
    # Use these to train with pre-trained weights
    
    # Define the normalization parameters for imagenet1k
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    
    # train_augmentations = transforms.Compose([transforms.Normalize(mean=mean, std=std),
    #                                         transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    #                                         transforms.ColorJitter(brightness=0.35, contrast=0.2, saturation=0.4, hue=0.1)])
    
    train_augmentations = transforms.Compose([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
                                            transforms.ColorJitter(brightness=0.35, contrast=0.2, saturation=0.4, hue=0.1)])
    # Set a seed for augmentations
    torch.manual_seed(42) 
    
    # Metrics collection for plotting
    train_losses = []
    train_f1_scores = []
    train_iou_scores = []
    
    val_losses = []
    val_f1_scores = []
    val_iou_scores = []
    
    best_val_loss = float('inf')

    # Train the model
    for epoch in range(num_epochs):
        train_loss = 0
        train_iou = 0
        train_f1 = 0
        
        val_iou = 0
        val_f1 = 0
        val_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            model.train()
            # Combine the inputs and targets into a single tensor
            data = torch.cat((inputs, targets), dim=1)
            # Apply the same augmentations to the combined tensor
            augmented_data = gt_augmentations(data)    
    
            # Split the augmented data back into individual inputs and targets
            inputs = augmented_data[:, :inputs.size(1)]
            targets = augmented_data[:, inputs.size(1):].to(device)

            inputs = train_augmentations(inputs).to(device)
      
            optimizer.zero_grad()
            outputs, eval_out = model(inputs)

            loss = criterion(outputs.to(device), targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            train_iou += iou_score(eval_out.to(device).detach(), targets)
            train_f1 += f1_score(eval_out.to(device).detach(),targets)
        
        if val_loader:
            model.eval()
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(val_loader): 
                
                    inputs, targets = inputs.to(device), targets.to(device)
                    logits,outputs = model(inputs)
                    
                    
                    val_loss = criterion(logits.to(device),targets)
                    val_loss += val_loss.item() * inputs.size(0)
                    
                    val_iou += iou_score(outputs.to(device), targets)
                    val_f1 += f1_score(outputs.to(device),targets)

                val_loss /= len(val_loader)
                val_iou /= len(val_loader)
                val_f1 /= len(val_loader)
        
        train_loss /= len(train_loader)
        train_iou /= len(train_loader)
        train_f1 /= len(train_loader)

    
        # Collect metrics for plotting
        train_losses.append(train_loss)
        train_f1_scores.append(train_f1.cpu().item())
        train_iou_scores.append(train_iou.cpu().item())
    
        if val_loader:
            val_losses.append(val_loss.cpu().item())
            val_f1_scores.append(val_f1.cpu().item())
            val_iou_scores.append(val_iou.cpu().item())
        
        # Check if currect val_loss is the best and save the weights
        if val_loader and val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the model weights
            if save_path:
                torch.save(model.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), '../models/best_pipeline.pth')
        
     # Print progress
        if lr_scheduler:
            print('Epoch: {} - Train Loss: {:.4f} - Learning Rate: {:.6f} - Train_IoU: {:.5f} - Train_F1: {:.5f}'.format(epoch+1, train_loss,optimizer.param_groups[0]['lr'], train_iou, train_f1))
            scheduler.step(val_loss)
            if val_loader:
                print('Val_F1: {:.5f}  - Val_IoU: {:.5f} '.format(val_f1,val_iou))
        else:
            print('Epoch: {} - Train Loss: {:.4f} - Train_IoU: {:.5f} - Train_F1: {:.5f}'.format(epoch+1, train_loss, train_iou, train_f1))
            if val_loader:
                print('Val_Loss: {} - Val_F1: {:.5f}  - Val_IoU: {:.5f} '.format(val_loss,val_f1,val_iou))
            
    if val_loader: 
        return train_losses,train_f1_scores,train_iou_scores,val_losses,val_f1_scores,val_iou_scores
    else:
        return train_losses,train_f1_scores,train_iou_scores 
    
if __name__ == '__main__':
    
    # Initialize ViT B-16
    vit_base = ViT(image_size=448, patch_size=16, num_classes=1, dim=768, depth=12, heads=12, 
                      mlp_dim=3072, dropout=0.1,load_pre= False)
    
    # vit_base.load_pretrained_weights(None)
    print(f'Number of trainable parameters for ViT : {vit_base.count_parameters()}')
    

    # Initialize MLP
    patch_classifier = DecoderMLP(n_classes = 1, d_encoder = 768, image_size=(448,448))
    print(f'Number of trainable parameters for MLP : {patch_classifier.count_parameters()}')
    
    segmenter = Segmenter(vit_base,patch_classifier,(448,448))
    print(f'Pipeline trainable params: {segmenter.count_parameters()}')
    