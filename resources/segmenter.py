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
from torch_poly_lr_decay import PolynomialLRDecay
import torchvision.transforms as transforms

# Set seed for randomize functions (Ez reproduction of results)
random.seed(1)

# Import TuSimple loader
import sys
sys.path.insert(0,'../resources/')
from tusimple import TuSimple
from mask_transformer import MaskTransformer
from vit import ViT
import utils
from linear import DecoderLinear

# Custom training function for the transformer pipeline with schedule and augmentations
def train(model, train_loader, val_loader = None, num_epochs=10, lr=0.01, weight_decay=0, lr_scheduler=True, lane_weight = None):
    # Set up loss function and optimizer
    criterion =  nn.BCEWithLogitsLoss(pos_weight= lane_weight)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    # Set up learning rate scheduler
    if lr_scheduler:
        scheduler = PolynomialLRDecay(optimizer, max_decay_steps=100, end_learning_rate=0.0001, power=0.9)

    # Set up device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    f1_score = F1Score(task="binary").to(device)
    iou_score = JaccardIndex(task= 'binary').to(device)
    
    train_augmentations = transforms.Compose([transforms.RandomRotation(degrees=(10, 30)),
                                         transforms.RandomHorizontalFlip()])
    
        
    # Set a seed for augmentations
    torch.manual_seed(42) 
    
    # Train the model
    for epoch in range(num_epochs):
        train_loss = 0
        train_iou = 0
        train_f1 = 0
        
        val_iou = 0
        val_f1 = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            model.train()
            # inputs, targets = inputs.to(device), targets.to(device)
            
            # Combine the inputs and targets into a single tensor
            data = torch.cat((inputs, targets), dim=1)
    
            
            # Apply the same augmentations to the combined tensor
            augmented_data = train_augmentations(data)    
    
            # Split the augmented data back into individual inputs and targets
            inputs = augmented_data[:, :inputs.size(1)].to(device)
            targets = augmented_data[:, inputs.size(1):].to(device)
      
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
                    outputs = model(inputs)
                
                    val_iou += iou_score(outputs.to(device), targets)
                    val_f1 += f1_score(outputs.to(device),targets)
        
                val_iou /= len(val_loader)
                val_f1 /= len(val_loader)
            
        train_loss /= len(train_loader)
        train_iou /= len(train_loader)
        train_f1 /= len(train_loader)
        
        
        
     # Print progress
        if lr_scheduler:
            print('Epoch: {} - Train Loss: {:.4f} - Learning Rate: {:.6f} - Train_IoU: {:.5f} - Train_F1: {:.5f}'.format(epoch+1, train_loss,optimizer.param_groups[0]['lr'], train_iou, train_f1))
            scheduler.step()
            if val_loader:
                print('Val_F1: {:.5f}  - Val_IoU: {:.5f} '.format(val_f1,val_iou))
        else:
            print('Epoch: {} - Train Loss: {:.4f} - Train_IoU: {:.5f} - Train_F1: {:.5f}'.format(epoch+1, train_loss, train_iou, train_f1))
            
# Segmenter pipeline class (ViT + Masks predictor end-to-end)
class Segmenter(nn.Module):
    def __init__(self,encoder, decoder, image_size = (640,640), output_act = 'sigmoid', find_threshold = False):
        super().__init__()
        self.patch_size = encoder.patch_size
        self.encoder = encoder
        self.decoder = decoder
        self.image_size = image_size
        self.lane_threshold = 0.5
        self.roc_flag = find_threshold
        if output_act == 'sigmoid':
            self.activation = nn.Sigmoid()   
        elif output_act == 'relu':
            self.activation = nn.ReLU()
        
        
    # Forward pass of the pipeline
    def forward(self, im):
        H, W = self.image_size
        
        # Pass through the pre-trained vit backbone
        x = self.encoder(im, return_features=True)
        
        # Pass through the masks transformer
        masks = self.decoder(x)
        

        # Interpolate patch level class annotatations to pixel level and transform to original image size
        masks = F.interpolate(masks, size=(H, W), mode="bilinear")
        
        # Training time
        if self.training:
            class_prob_masks = self.activation(masks)
            predictions = torch.where(class_prob_masks > self.lane_threshold, torch.ones_like(class_prob_masks), torch.zeros_like(class_prob_masks))
            return masks, predictions
        
        elif self.roc_flag and not self.training:
            class_prob_masks = self.activation(masks)
            return class_prob_masks
        else:
            class_prob_masks = self.activation(masks)
            predictions = torch.where(class_prob_masks > self.lane_threshold, torch.ones_like(class_prob_masks), torch.zeros_like(class_prob_masks))
            return predictions
        
    # Count pipeline trainable parameters
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    # Load trained model
    def load_segmenter(self,path):    
        self.load_state_dict(torch.load(path,map_location=torch.device('cpu')))