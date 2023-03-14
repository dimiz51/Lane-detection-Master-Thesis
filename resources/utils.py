from torchvision import transforms
import numpy as np
import cv2
import torch
    
    
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
    
# Calculate batch mean IoU (Intersection over Union) metric (used in loops with batches )
def mean_iou(pred, target):
    preds = torch.round(pred).long()
    targets = torch.round(target).long()
    
    # loop through batch dimension
    iou = 0
    for i in range(pred.shape[0]):
        intersection = torch.logical_and(preds, targets).sum(dim=(1, 2))
        union = torch.logical_or(preds, targets).sum(dim=(1, 2))
        iou = (intersection + 1e-10) / (union + 1e-10)
    return iou.mean().item()

# Calculate mean batch accuracy metric (used in loops with batches)
def accuracy(pred, target):
    # loop through batch dimension
    acc = 0
    for i in range(pred.shape[0]):
        acc += (pred[i] == target[i]).float().mean()
        
    return acc.item() / pred.shape[0]