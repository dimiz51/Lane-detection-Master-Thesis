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
    
# Return the confusion matrix
def confmat (pred,target):
    confmat = ConfusionMatrix(task="binary", num_classes=2)
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


# Dice loss function
def dice_loss(outputs, targets, smooth=1e-7):
    intersection = (outputs * targets).sum()
    union = outputs.sum() + targets.sum()
    dice = (2 * intersection + smooth) / (union + smooth)
    loss = 1 - dice
    return loss


# ROC AUC Curve to find binary mask optimal threshold
def find_threshold(prob_preds,gt_mask):
    
    prob_arr = prob_preds.detach().cpu().numpy()
    gt_arr = gt_mask.detach().cpu().numpy()
    
    
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(gt_arr.ravel(),prob_arr.ravel())
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
    # Find best threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    
    # Print best threshold
    print(f"The optimal threshold is: {optimal_threshold}")
    
    return optimal_threshold

# Get ROC Curve inputs from test set (needs a trained model)
def get_roc_probs(model,test_loader):      
    batch_probs = []
    batch_gt = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
    if model.roc_flag:
        model.eval()
        with torch.no_grad():  # disable gradient calculation during inference
            for batch_idx, (inputs, targets) in enumerate(test_loader): 
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                batch_probs.append(outputs)
                batch_gt.append(targets)
            
        pred_probs = torch.cat(batch_probs, dim=0)
        ground_truths = torch.cat(batch_gt, dim=0)
        return pred_probs, ground_truths
    else:
        return 1