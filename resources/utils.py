from torchvision import transforms
import numpy as np
import cv2
import torch
from torchmetrics import ConfusionMatrix
from einops import rearrange
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
    
    
# Helper func to transform back to array from tensor
def toImagearr(img_tens):
    convert = transforms.Compose([transforms.ToPILImage()])
    im_array = np.array(convert(img_tens))
    return im_array

# Helper func to display image with OpenCV
def disp_img(image: np.array , name = 'Image'):
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def save_img(image: np.array , name = 'Image', save = False):
    #save the image as well
    if save:
        cv2.imwrite(name + ".jpg", image)
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
# Return the confusion matrix
def confmat (pred,target):
    confmat = ConfusionMatrix(task="binary")
    return confmat(pred,target)

# Helper func to plot image and ground truth simultaneously
def plot_img_pred(tensor, pred_mask):
    img = toImagearr(tensor)
    rgb_tensor = torch.stack((pred_mask,)*3, dim=1).squeeze(0) 
    pred_mask = toImagearr(rgb_tensor)
    hori = np.concatenate((img, pred_mask), axis=1)
    disp_img(hori,'Image/Predicted Mask')
    

# Calculate lane class weights for the training set
def calculate_class_weight (train_set):
    lane_pixels = 0
    back_pixels = 0
    
    for img,gt in train_set:
        lane_pixels += (gt == 1.).sum()
        back_pixels += (gt == 0.).sum()
    
    pos_weight = back_pixels / lane_pixels
    
    return pos_weight.int()