from torchvision import transforms
import numpy as np
import cv2
import torch
from torchmetrics import ConfusionMatrix
    
    
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
    
# Calculate mean batch accuracy metric (used in loops with batches)
def accuracy(pred, target):
    # loop through batch dimension
    acc = 0
    for i in range(pred.shape[0]):
        acc += (pred[i] == target[i]).float().mean()
        
    return acc.item() / pred.shape[0]

# Return the confusion matrix
def confmat (pred,target):
    confmat = ConfusionMatrix(task="binary", num_classes=2)
    return confmat(pred,target)