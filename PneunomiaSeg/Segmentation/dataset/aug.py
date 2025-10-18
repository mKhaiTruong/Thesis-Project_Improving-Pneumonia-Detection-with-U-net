import cv2
import sys
import numpy as np
from glob import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2

def trainAug(img_size):
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.3)
        ], p=0.3),
        
        # A.OneOf([
        #     A.GridDropout(ratio=0.2, unit_size_min=2, unit_size_max=4, random_offset=True, p=0.2),
        # ], p=0.3),
        
        # A.OneOf([
        #     A.GaussianBlur(blur_limit=(2, 2), p=0.2),   
        #     A.GaussNoise(var_limit=(5.0, 10.0), p=0.2), 
        # ], p=0.3),
        
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(p=1.0),
    ])
    

def validAug(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(p=1.0),
    ])
    

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def reverse_normalize_tensor(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    
    if tensor.ndim == 4:
        tensor = tensor[0]  

    img = tensor.clone().detach().cpu()

    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)

    img = img.permute(1, 2, 0).numpy()   # [H,W,C]
    img = np.clip(img, 0, 1)

    return img