# Imports

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision.transforms import ToPILImage
from torchvision.transforms import InterpolationMode
import json
import numpy as np
import random
import cv2

# Set seed for randomize functions (Ez reproduction of results)
random.seed(100)

# Define a custom dataset class for our data splits
class BaseSplitClass(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# TuSimple Dataset loader and pre-processing class
# Full Size: Train(3626 clips/ 20 frames per clip/ 20th only is annotated), Test(2782 clips/ 20 frames per clip/ 20th only annotated)
# Link: https://github.com/TuSimple/tusimple-benchmark/tree/master/doc/lane_detection
class TuSimple(Dataset):  
    def __init__(self, train_annotations : list, train_img_dir: str, resize_to : tuple , subset_size = 0.2, image_size = (1280,720), val_size = 0):
        self.images_size = image_size
        self.resize = resize_to
        self.subset = subset_size
        self.train_dir = train_img_dir
        self.complete_gt = train_annotations
        self.complete_size = len(train_annotations)
        self.sf_w = round(resize_to[1] / 1280, 4)
        self.sf_h = round(resize_to[0] / 720, 4)
        self.val_size = val_size
        self.train_dataset, self.train_gt = self.generate_dataset()


    def __len__(self):
        if len(self.train_dataset) == len(self.train_gt):
            return len(self.train_gt)
        else:
            return "Dataset generation failure: Size of training images does not match the existing ground truths."
    
    # return element from the trainset
    def __getitem__(self, idx):
        if len(self.train_dataset) == len(self.train_gt):
            img_tensor = self.train_dataset[idx]
            img_gt = self.train_gt[idx]
            return img_tensor, img_gt
        else:
            return "The dataset hasn't been constructed properly. Generate again!"
    
    # Generate segmentation mask for a given image NOTE: np.array image dims = (H,W,C)
    def generate_seg_mask(self,ground_truth: dict):
        image_path = ground_truth['raw_file']
        image = cv2.imread(os.path.join(self.train_dir,image_path))
        
        nolane_token = -2 
        h_vals = ground_truth['h_samples']
        lanes = ground_truth['lanes']
        lane_val = 255
        
        lane_markings_list = []
        for lane in lanes:
            x_coords = []
            y_coords = []
            for i in range(0,len(lane)):             
                if lane[i] != nolane_token:
                    x_coords.append(lane[i])
                    y_coords.append(h_vals[i])
                    lane_markings = list(zip(x_coords, y_coords))
            lane_markings_list.append(lane_markings)  
        
        # Find resized lane anchor points
        resized_lanes = []
    
        for lane in lane_markings_list:
            resized_lane = []
            for c in lane:
                new_c = (int(c[0] * self.sf_w), int(c[1] * self.sf_h))
                resized_lane.append(new_c)
            resized_lanes.append(resized_lane)
        
        # Create empty black mask for ground truth
        resized_mask = np.zeros(self.resize,dtype= np.uint8)
        
        # loop through the lane points and draw thickened white polylines for each lane
        for lane_points_resized in resized_lanes:
            cv2.polylines(resized_mask, [np.array(lane_points_resized)], isClosed=False, color=(255, 255, 255), thickness=5)
              
        return resized_mask  


    # Returns original image size for the dataset    
    def get_original_size(self):
        return self.images_size
    
    # Helper func to display image with OpenCV
    def disp_img(self, image: np.array , name = 'Image'):
        cv2.imshow(name,image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    # Helper func to transform back to array from tensor
    def toImagearr(self,img_tens):
        convert = transforms.Compose([transforms.ToPILImage()])
        im_array = np.array(convert(img_tens))
        return im_array
    
    # Helper func to plot image and ground truth simultaneously
    def plot_img_gt(self, tensor, gt_mask):
        img = self.toImagearr(tensor) 
        rgb_tensor = torch.stack((gt_mask,)*3, dim=1).squeeze(0)
        gt_mask = self.toImagearr(rgb_tensor)
        Hori = np.concatenate((img, gt_mask), axis=1)
        self.disp_img(Hori,'Image/Ground Truth')
               
    # Get list of lists containing ground truth lane pixel values for all lanes with respect to the original number of lanes in the original gt
    def get_resized_gt(self, original_gt: dict, new_size = tuple):
        seg_gt_mask = self.generate_seg_mask(original_gt)
        
        seg_gt_mask = Image.fromarray(np.uint8(seg_gt_mask))
        
        gt_transforms = transforms.Compose([transforms.ToTensor()])
    
        resized_gt_tensor = gt_transforms(seg_gt_mask)

        new_gt = resized_gt_tensor.float()
        
        return new_gt
              
    # Partition dataset according to input subset size and dynamically generate the train/val splits
    def generate_dataset(self):
        train_set = []
        
        complete_idx = [idx for idx in range(0, self.complete_size)]
        target_samples = int(self.complete_size * self.subset)
        shuffled = random.sample(complete_idx,len(complete_idx))

        # Pick n (target samples no) idx from the shuffled dataset
        dataset_idxs = [shuffled[idx] for idx in range(0, target_samples)]
        train_gt = [self.complete_gt[idx] for idx in dataset_idxs]
        
        resized_train_gt = [self.get_resized_gt(ground,self.resize) for ground in train_gt]
        
        # Load images, resize inputs, generate resized ground truth seg masks,transform to tensors and generate dataset (or subset)
        for gt in train_gt:
            img_path = gt['raw_file']
            train_transforms = transforms.Compose([transforms.Resize(size = self.resize,interpolation=InterpolationMode.BICUBIC),
                                                   transforms.ToTensor()
                                                   ])
            image = cv2.imread(os.path.join(self.train_dir, img_path))
            image = Image.fromarray(np.uint8(image))
            img_tensor = train_transforms(image)
            train_set.append(img_tensor)
        
        return train_set, resized_train_gt   
    
    # Generate train and validation splits dynamically (after this operation use del dataset to free memory)
    def train_val_split(self):
        X = self.train_dataset
        Y = self.train_gt
        
        # Split the generated train set into train and val
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size= self.val_size,random_state=42)
        
        train_set = BaseSplitClass(X_train, Y_train)
        validation_set = BaseSplitClass(X_val,Y_val)
        
        return train_set, validation_set
        
        

