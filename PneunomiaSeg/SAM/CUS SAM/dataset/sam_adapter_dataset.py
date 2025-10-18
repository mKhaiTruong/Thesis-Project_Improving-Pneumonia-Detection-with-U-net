import os, cv2, sys
from PIL import Image, ImageOps
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(r"..\PneunomiaSeg\SAM\CUS SAM")
from dataset.aug import train_transforms
from sam_helpers.sam_helpers import *
from utils.visualization import *
from configuration.covid_ct import Covid_CT_Scan_Config
from configuration.covid_ct_2 import Covid_CT_Scan_2_Config
from configuration.covid_ct_3 import Covid_CT_Scan_3_Config
from configuration.covid_lesion import Covid_Lesion_Config



class TestingDataset(Dataset):
    
    def __init__(self, images, masks, requires_name=True, return_og_msk=False,
                 all_classes=None, desired_classes=None, image_size=256,
                 point_num=1):
        
        self.image_paths = images
        self.mask_paths  = masks
        
        self.image_size  = image_size
        self.pixels_mean = [123.675, 116.28, 103.53 ]
        self.pixels_std  = [ 58.395,  57.12,  57.375]
        self.point_num   = point_num
        
        self.requires_name = requires_name
        self.return_og_msk = return_og_msk
        self.classes     = desired_classes
        self.all_classes = all_classes
        
        if self.classes:
            self.class_ids = [all_classes.index(c) for c in self.classes]
    
    
    def _read_img(self, img_pth):
        img = cv2.imread(img_pth, cv2.IMREAD_COLOR)             # BGR
        if img is None:
            raise ValueError(f"Cannot read image: {img_pth}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        img = np.array(img).astype(np.float32)
        
        img = (img - self.pixels_mean) / self.pixels_std
        return img
    
    
    def _read_msk(self, msk_pth):
        
        msk = cv2.imread(msk_pth, 0)
        if msk is None:         raise ValueError(f"Cannot read mask: {msk_pth}")
        if msk.max() == 255:    msk = msk / 255
        
        # Binary mask 0/1
        msk = (msk > 0).astype(np.uint8)
        msk = cv2.resize(msk, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        # Multi-class mask
        if self.classes:
            msk = np.isin(msk, self.class_ids).astype(np.uint8)
            
        return msk
        
        
    def __getitem__(self, index):
        
        input = {}
        
        # ---- Img & Msk ----
        og_np_img = self._read_img(self.image_paths[index])
        og_np_msk = self._read_msk(self.mask_paths[index])
        
        # ---- AUG ----
        ori_mask = torch.tensor(og_np_msk).unsqueeze(0)

        h, w = og_np_msk.shape
        transforms = train_transforms(self.image_size, h, w)
        augments = transforms(image=og_np_img, mask=og_np_msk)
        img, msk = augments['image'], augments['mask'].to(torch.int64)
        
        # >> PROMPTS
        boxes = get_boxes_from_mask(msk, max_pixel = 0)
        point_coords, point_labels = init_point_sampling(msk, self.point_num)
        
        # >> INPUT
        input["image"] = img
        input["label"] = msk.unsqueeze(0)
        input["point_coords"]  = point_coords
        input["point_labels"]  = point_labels
        input["boxes"] = boxes
        input["original_size"] = (h, w)
        
        image_name = self.image_paths[index].split('\\')[-1]
        if self.requires_name:
            input['name'] = image_name
        
        if self.return_og_msk:
            input['ori_label'] = ori_mask
    
        return input
    
    def __len__(self): return len(self.image_paths) 
    
    def _denormalize(self, img):
        img = img * self.pixels_std + self.pixels_mean
        return img


class TrainingDataset(Dataset): 
    
    def __init__(self, images, masks, requires_name=True,
                 all_classes=None, desired_classes=None, 
                 image_size=512, point_num=1, mask_num=1):
        self.image_size     = image_size
        self.requires_name  = requires_name
        self.point_num      = point_num
        self.mask_num       = mask_num
        
        self.pixels_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.pixels_std  = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        
        self.image_paths = images
        self.mask_paths  = masks
        self.classes     = desired_classes
        self.all_classes = all_classes
        
        if self.classes:
            self.class_ids = [all_classes.index(c) for c in self.classes]
    
    def _read_img(self, img_pth):
        img = cv2.imread(img_pth, cv2.IMREAD_COLOR)             # BGR
        if img is None:
            raise ValueError(f"Cannot read image: {img_pth}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img).astype(np.float32)
        img = (img - self.pixels_mean) / self.pixels_std
        return img
    
    def _read_msk(self, msk_pth):
        msk = cv2.imread(msk_pth, cv2.IMREAD_GRAYSCALE)
        if msk is None:
            raise ValueError(f"Cannot read mask: {msk_pth}")
        
        # Binary mask 0/1
        msk = (msk > 0).astype(np.uint8)

        # Multi-class mask
        if self.classes:
            msk = np.isin(msk, self.class_ids).astype(np.uint8)
            
        return msk
    
    
    def __getitem__(self, index):
        
        input = {}
        
        img = self._read_img(self.image_paths[index])    # ---- Image ----
        msk = self._read_msk(self.mask_paths[index])     # ---- Mask ----
        
        # ---- AUG ----  
        h, w, _ = img.shape
        transforms = train_transforms(self.image_size, h, w)
        
        masks_list, boxes_list, pts_list, lbs_list = [], [], [], []
        for _ in range(self.mask_num):
            aug = transforms(image=img, mask=msk)
            
            img_t, msk_t = aug['image'], aug['mask'].to(torch.int64)
            boxes_t      = get_boxes_from_mask(msk_t)
            pts_t, lbs_t = init_point_sampling(msk_t, self.point_num)
            
            masks_list.append(msk_t)
            boxes_list.append(boxes_t)
            pts_list.append(pts_t)
            lbs_list.append(lbs_t)
        
        
        input['image']        = img_t.unsqueeze(0)                      # [1,C,H,W]
        input['label']        = torch.stack(masks_list,0).unsqueeze(1)  # [K,1,H,W]
        input['boxes']        = torch.stack(boxes_list,0)               # [K,B,4]
        input['point_coords'] = torch.stack(pts_list,0)                 # [K,P,2]
        input['point_labels'] = torch.stack(lbs_list,0)                 # [K,P]         
        
        image_name = self.image_paths[index].split('\\')[-1]
        if self.requires_name:
            input['name'] = image_name
            
        return input
    
    def __len__(self): return len(self.image_paths)
    
    
if __name__ == '__main__':
    
    config = Covid_CT_Scan_Config()
    all_labels, desired_labels = config.all_labels, config.only_labels
    
    trainImageList = config.trainImagesDir
    trainMaskList  = config.trainMasksDir
    
    for label in desired_labels:
        
        dataset = TrainingDataset(
            trainImageList, trainMaskList, 
            all_classes=all_labels, desired_classes=[label], mask_num=2)
        # dataset = TestingDataset(
        #     trainImageList, trainMaskList, 
        #     all_classes=all_labels, desired_classes=[label], point_num=1)
        print("Dataset: ", len(dataset))
        
        train_batch_sampler = DataLoader(
            dataset, batch_size=2, shuffle=True, num_workers=4)
        
        # SUMMARY: SIZE OF DATASET AND DATALOADER
        '''
        C: channels             (image = 3 and mask = 1)
        K: number of labels
        H, W: height, width
        B: number of boxes
        P: number of points
        BS: batch size
        
        **Attention**: 
            + In theory, we should expect K = B = P = number of point_labels
            + Boxes have 4 attributes: (X1, X2, Y1, Y2)
            + Points have 2 attributes: (X, Y)
        
        DATASET:
            + img:      torch.Size( [1, C, H, W] )
            + msk:      torch.Size( [K, C, H, W] )
            + boxes:    torch.Size( [K, B, 4] )
            + points:   torch.Size( [K, P, 2] )
            + point_labels: torch.Size( [K, P] )
        
        DATALOADER:
            + img:      torch.Size( [BS, C, H, W] )
            + msk:      torch.Size( [K*BS, C, H, W] )
            + boxes:    torch.Size( [K*BS, B, 4] )
            + points:   torch.Size( [K*BS, P, 2] )
            + point_labels: torch.Size( [K*BS, P] )
        '''
        
        j = 0
        for i, batch in enumerate(tqdm(train_batch_sampler)):
            batched_images = stack_dict_batch(batch)
            
            # print(batched_images['image'].shape, batched_images['label'].shape)
            # print(batched_images['boxes'].shape)
            # print(batched_images['point_coords'].shape, batched_images['point_labels'].shape)
            # print()
            
            # ----------- Visualize sample -----------
            img_t   = batched_images['image'][0]           # [C,H,W]
            mask_t  = batched_images['label'][0,0]         # [H,W]
            boxes_t = batched_images['boxes'][0]           # [B,4]
            pts_t   = batched_images['point_coords'][0]    # [P,2]
            lbs_t   = batched_images['point_labels'][0]    # [P]

            visualize_sample(img_t, mask_t)
            
            j += 1
            if j == 10:
                break

            