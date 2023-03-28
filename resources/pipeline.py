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
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import F1Score,JaccardIndex
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torch_poly_lr_decay import PolynomialLRDecay
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# Import TuSimple loader
import sys
sys.path.insert(0,'../resources/')
from tusimple import TuSimple
from vit import ViT
from mlp_decoder import DecoderMLP
from segnet_backbone import SegNet
import utils

# End-to-End pipeline (CNN + ViT + MLP)
class Pipeline(nn.Module):
    def __init__(self, feat_extractor, vit_variant, mlp_head, image_size = (448,448)):
        super().__init__()
        self.cnn = feat_extractor
        self.transformer = vit_variant
        self.mlp = mlp_head
        self.image_size = image_size
        self.lane_threshold = 0.5
        self.activation = nn.Sigmoid()
        
    # Forward pass of the pipeline
    def forward(self, im):
        H, W = self.image_size
        
        # CNN branch for feature extraction
        x,_ = self.cnn(im)
        
        # Standardize featmaps to prevent exploding gradients and help the ViT perform better
        x = self.standarize_layer(x)
        
        # Transform standardized feature maps using the ViT
        x = self.transformer(x)
        
        # Perform patch level classification (0 for background/ 1 for lane)
        x = self.mlp(x)

        # Interpolate patch level class annotatations to pixel level and transform to original image size
        logits = F.interpolate(x, size=(H, W), mode="bilinear")
        
        probs = self.activation(logits)

        return logits, probs
        
        
    # Standardize feature maps from SegNet layer
    def standarize_layer(self,featmaps):
        # Compute mean and standard deviation of each channel
        mean = torch.mean(featmaps, dim=[0, 2, 3], keepdim=True)
        std = torch.std(featmaps, dim=[0, 2, 3], keepdim=True)

        # Normalize each channel to have zero mean and unit variance
        normal_featmaps = (featmaps - mean) / std
        return normal_featmaps
        
    # Count pipeline trainable parameters
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    # Load trained model
    def load_weights(self,path): 
        self.load_state_dict(torch.load(path,map_location=torch.device('cpu')))

    # Custom training function for the pipeline with schedule and augmentations
def train(model, train_loader, val_loader = None, num_epochs=10, lr=0.01, weight_decay=0, SGD_momentum = 0.9, lr_scheduler=False, lane_weight = None):
    # Set up loss function and optimizer
    criterion =  nn.BCEWithLogitsLoss(pos_weight= lane_weight)

    #create seperate parameter groups for the different networks
    segnet_params = [p for p in model.cnn.parameters() if p.requires_grad]
    mlp_params = [p for p in model.mlp.parameters() if p.requires_grad]
    vit_params = [p for p in model.transformer.parameters() if p.requires_grad]

    #define the optimizer with different learning rates for the parameter groups
    optimizer = optim.SGD([
        {'params' : segnet_params, 'lr' : 0.01},
        {'params' : mlp_params, 'lr' : 0.01},
        {'params' : vit_params, 'lr' : 0.001}
    ], momentum=SGD_momentum)
    
    # Define your learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=4, verbose=True)
    # Set up learning rate scheduler
    if lr_scheduler:
        pass

    # Set up device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    f1_score = F1Score(task="binary").to(device)
    iou_score = JaccardIndex(task= 'binary').to(device)
    
    gt_augmentations = transforms.Compose([transforms.RandomRotation(degrees=(10, 30)),
                                              transforms.RandomHorizontalFlip()])
  
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

        scheduler.step(val_loss)
    
        # Collect metrics for plotting
        train_losses.append(train_loss)
        train_f1_scores.append(train_f1)
        train_iou_scores.append(train_iou)
    
        if val_loader:
            val_losses.append(val_loss)
            val_f1_scores.append(val_f1)
            val_iou_scores.append(val_iou)
        
        
     # Print progress
        if lr_scheduler:
            print('Epoch: {} - Train Loss: {:.4f} - Learning Rate: {:.6f} - Train_IoU: {:.5f} - Train_F1: {:.5f}'.format(epoch+1, train_loss,optimizer.param_groups[0]['lr'], train_iou, train_f1))
            # scheduler.step()
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
    
