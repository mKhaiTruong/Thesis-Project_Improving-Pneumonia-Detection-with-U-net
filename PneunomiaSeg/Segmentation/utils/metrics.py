import torch

def compute_iou(preds, masks, threshold=0.5, eps=1e-6):
    
    preds_bin = (preds > threshold).float()
    
    intersection = (preds_bin * masks).sum(dim=(1,2,3))
    union        = ((preds_bin + masks) > 0).float().sum(dim=(1,2,3))
    iou          = (intersection + eps) / (union + eps)
    return iou.mean().item()