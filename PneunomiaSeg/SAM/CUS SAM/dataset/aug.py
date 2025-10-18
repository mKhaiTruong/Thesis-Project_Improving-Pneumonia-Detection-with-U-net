from albumentations.pytorch import ToTensorV2
import cv2, os, random, logging
import numpy as np
import albumentations as A
from matplotlib import pyplot as plt

def train_transforms(img_size, og_h, og_w):
    transforms = []
    
    if og_h < img_size and og_w < img_size:
        transforms.append(
            A.PadIfNeeded(min_height=img_size, min_width=img_size, 
                          border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0)
        )
    else:
        transforms.append(
            A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR)
        )
    
    transforms.append(ToTensorV2(p=1.))
    return A.Compose(transforms, p=1.)