# Imports

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from torchvision.transforms import ToPILImage
import json
import numpy as np
import random
import cv2

# Set seed for randomize functions (Ez reproduction of results)
random.seed(100)


# TuSimple Dataset loader and pre-processing class
# Full Size: Train(3626 clips/ 20 frames per clip/ 20th only is annotated), Test(2782 clips/ 20 frames per clip/ 20th only annotated)
# Link: https://github.com/TuSimple/tusimple-benchmark/tree/master/doc/lane_detection
class TuSimple(Dataset):  
    def __init__(self, train_annotations : list, train_img_dir: str, resize_to : tuple , subset_size = 0.2, image_size = (1280,720), val_size = 0.15):
        self.images_size = image_size
        self.resize = resize_to
        self.val_size = val_size
        self.subset = subset_size
        self.train_dir = train_img_dir
        self.complete_gt = train_annotations
        self.complete_size = len(train_annotations)
        self.train_dataset, self.train_gt = self.generate_dataset()
        
    def __len__(self):
        if len(self.train_dataset) == len(self.train_gt):
            return len(self.train_gt)
        else:
            return "Dataset generation failure: Size of training images does not match the existing ground truths."
    
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
        
        masks = np.zeros_like(image[:,:,0])
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
        for z in lane_markings_list:
            for x,y in z:
                masks[y,x] = 1
        seg_mask = cv2.bitwise_and(image, image, mask=masks)
        seg_mask [seg_mask != 0] = lane_val
        return seg_mask  


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
        Hori = np.concatenate((img, gt_mask), axis=1)
        self.disp_img(Hori,'Image/Ground Truth')
               
    # Get list of lists containing ground truth lane pixel values for all lanes with respect to the original number of lanes in the original gt
    def get_resized_gt(self, original_gt: dict, new_size = tuple):
        seg_gt_mask = self.generate_seg_mask(original_gt)
        resized_gt_mask = cv2.resize(seg_gt_mask, new_size, interpolation = cv2.INTER_LINEAR)
                
        # Set all resized pixels color to white (thresholding)
        resized_gt_mask [resized_gt_mask !=0] = 255
        
        gt_transforms = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize(size = self.resize)])
        
        resized_gt_tensor = gt_transforms(seg_gt_mask)
        
        new_gt = {'ground_truth_mask': resized_gt_mask,'gt_tensor': resized_gt_tensor,'raw_file': original_gt['raw_file']}
        
        return new_gt
              
    # Partition dataset according to input subset size and dynamically generate the train/val splits
    def generate_dataset(self):
        train_set = []
        
        complete_idx = [idx for idx in range(0, self.complete_size + 1)]
        target_samples = int(self.complete_size * self.subset)
        # val_samples = int(len(target_samples) * self.val_size)
        shuffled = random.sample(complete_idx,len(complete_idx))
        
        # Pick n (target samples no) idx from the shuffled dataset
        dataset_idxs = [shuffled[idx] for idx in range(0, target_samples)]
        train_gt = [self.complete_gt[idx] for idx in dataset_idxs]
        
        resized_train_gt = [self.get_resized_gt(ground,self.resize) for ground in train_gt]
        
        # Load images, resize inputs, generate resized ground truth seg masks,transform to tensors and generate dataset (or subset)
        for gt in train_gt:
            img_path = gt['raw_file']
            train_transforms = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Resize(size = self.resize)])
            image = cv2.imread(os.path.join(self.train_dir, img_path))
            
            img_tensor = train_transforms(image)
            train_set.append(img_tensor)
        
        return train_set, resized_train_gt   
        



def main():
    
    # ROOT DIRECTORIES
    root_dir = os.path.dirname(os.getcwd())
    annotated_dir = os.path.join(root_dir,'datasets/tusimple/train_set/annotations')
    clips_dir = os.path.join(root_dir,'datasets/tusimple/train_set/')
    annotated = os.listdir(annotated_dir)
    
    # Get path directories for clips and annotations for the TUSimple dataset + ground truth dictionary
    annotations = list()
    for gt_file in annotated:
        path = os.path.join(annotated_dir,gt_file)
        json_gt = [json.loads(line) for line in open(path)]
        annotations.append(json_gt)
    
    annotations = [a for f in annotations for a in f]
    
    dataset = TuSimple(train_annotations = annotations, train_img_dir = clips_dir, resize_to = (640,640), subset_size = 0.05)
    
    img_tns, gt = dataset[0]
    dataset.plot_img_gt(img_tns,gt['ground_truth_mask'])
    
main()