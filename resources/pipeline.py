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
        
        # Transform standardized feature maps using the ViT
        x = self.transformer(x)
        
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
    segnet_params = [p for p in model.cnn.parameters() if p.requires_grad]
    mlp_params = [p for p in model.mlp.parameters() if p.requires_grad]
    vit_params = [p for p in model.transformer.parameters() if p.requires_grad]

    #define the optimizer with different learning rates for the parameter groups
    optimizer = optim.SGD([
        {'params' : segnet_params, 'lr' : lr},
        {'params' : mlp_params, 'lr' : lr},
        {'params' : vit_params, 'lr' : lr}
    ], momentum=SGD_momentum, weight_decay= weight_decay)
    
    # Define your learning rate scheduler
    if lr_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True, threshold=1e-4, min_lr=[0.0001,0.0001, 0.001])

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
    # Initialize SegNet with pre-trained frozen weights
    cnn = SegNet()
    cnn.load_weights('../models/best_segnet.pth')
    cnn.freeze_all_but_some([])
    print(f'Number of trainable parameters for SegNet : {cnn.count_parameters()}')

    # Initialize ViT Tiny
    vit_tiny = ViT(image_size=448, patch_size=16, num_classes=1, dim=192, depth=6, heads=3, 
                      mlp_dim=768, dropout=0.1,load_pre= False)
    print(f'Number of trainable parameters for ViT : {vit_tiny.count_parameters()}')

    # Initialize MLP
    patch_classifier = DecoderMLP(n_classes = 1, d_encoder = 192, image_size=(448,448))
    print(f'Number of trainable parameters for MLP : {patch_classifier.count_parameters()}')
    
    # Create pipeline instance
    model = Pipeline(cnn, vit_tiny, patch_classifier, image_size= (448,448))
    print(f'Number of trainable parameters for Pipeline : {model.count_parameters()}')
    
    # ROOT DIRECTORIES
    root_dir = os.path.dirname(os.getcwd())
    annotated_dir = os.path.join(root_dir,'datasets/tusimple/train_set/annotations')
    clips_dir = os.path.join(root_dir,'datasets/tusimple/train_set/')
    annotated = os.listdir(annotated_dir)

    annotations = list()
    for gt_file in annotated:
        path = os.path.join(annotated_dir,gt_file)
        json_gt = [json.loads(line) for line in open(path)]
        annotations.append(json_gt)
    
    annotations = [a for f in annotations for a in f]
    
    dataset = TuSimple(train_annotations = annotations, train_img_dir = clips_dir, resize_to = (448,448), subset_size = 1, val_size= 0.15)
    train_set, validation_set = dataset.train_val_split()
    del dataset
    
    # Lane weight
    pos_weight = utils.calculate_class_weight(train_set)
    
    # Create dataloaders for train and validation 
    train_loader = DataLoader(train_set, batch_size= 4,shuffle= True, drop_last= True, num_workers= 8) 
    validation_loader = DataLoader(validation_set,batch_size= 4, shuffle= True, drop_last= True, num_workers= 8) 
    
    # Train the model
    train_losses,train_f1_scores,train_iou_scores,val_losses,val_f1_scores,val_iou_scores = train(model, train_loader,val_loader= validation_loader , num_epochs= 100, 
                                                                                        lane_weight = pos_weight, lr = 0.01, SGD_momentum= 0.9, lr_scheduler= True)
    
    # Plot metrics after training for train and validation sets (bins of 5 epochs)
    plot_metrics(train_losses,val_losses,train_f1_scores,val_f1_scores,train_iou_scores,val_iou_scores)