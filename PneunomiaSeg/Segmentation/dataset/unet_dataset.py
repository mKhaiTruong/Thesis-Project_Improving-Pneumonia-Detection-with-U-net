import os, cv2, sys, pydicom
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset

sys.path.append(r"..\PneunomiaSeg\SAM\CUS SAM")
from dataset.aug import *
from configuration.covid_ct import Covid_CT_Scan_Config
from configuration.covid_ct_2 import Covid_CT_Scan_2_Config
from configuration.covid_ct_3 import Covid_CT_Scan_3_Config
from configuration.covid_lesion import Covid_Lesion_Config

from utils.visualization import *

class Test_UnetDataset(Dataset):
    def __init__(self, img_paths=None, img_size=256):
        self.img_paths = img_paths
        self.img_size  = img_size

        self.transform = validAug(img_size)
    
    def __len__(self):  return len(self.img_paths)
    
    def _load_image(self, path):
        ext = os.path.splitext(path)[-1].lower()

        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            img = Image.open(path).convert("RGB")
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
            img = np.array(img)
        
        elif ext == '.tif' or ext == '.tiff':
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size))
        
        elif ext == '.dcm':
            dcm = pydicom.dcmread(path)
            img = dcm.pixel_array.astype(np.float32)

            # normalize 0-255
            img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-5)
            img = (img * 255).astype(np.uint8)
            
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size))

        else:
            raise ValueError(f"Unsupported image format: {ext}")
        
        return img
    
    def __getitem__(self, idx=0, img_in=None):
        img_pth = self.img_paths[idx]
        img = self._load_image(img_pth)
        
        if img_in is not None and isinstance(img_in, Image.Image):
            img = np.array(img_in)
            
        img = self.transform(image=img)['image']
        
        return {
            'image': img,
            'name' : os.path.basename(img_pth)
        }


class UnetDataset(Dataset): 
    
    def __init__(self, images, isTrain=False, masks=None, requires_name=False, 
                 image_size=256, all_classes=None, desired_classes=None, cache_size=256, 
                 domain_based = False
                 ):
        
        '''
        + isTrain and masks: Training mode
        + !isTrain and masks: Validating mode
        + !isTrain and !masks: Testing/Inference mode
        
        + isTrain and !masks: Error - Training requires masks
        '''
        
        if isTrain and masks is None:
            raise ValueError("Training dataset requires masks, but masks=None was provided.")
        
        self.image_paths = images
        self.mask_paths  = masks
        self.isTrain     = isTrain
        self.image_size  = image_size
        
        self.requires_name = requires_name or not isTrain
        self.transforms = trainAug(image_size) if isTrain else validAug(image_size)

        self._cache = {}
        self._cache_size = cache_size
        
        self.domain_based= domain_based
        self.all_classes = all_classes
        self.classes     = desired_classes
        if self.classes:
            self.class_ids = [self.all_classes.index(c) for c in self.classes]
        
        # Sanity check
        assert len(self.image_paths) > 0, "No images found"
        if masks:
            assert len(self.image_paths) == len(self.mask_paths), \
                f"Number of images ({len(self.image_paths)}) != number of masks ({len(self.mask_paths)})"

        
    def _read_image(self, img_pth):
        img = cv2.imread(img_pth, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            raise ValueError(f"Cannot read image: {img_pth}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        return img


    def _read_mask(self, msk_pth):
        msk = cv2.imread(msk_pth, cv2.IMREAD_GRAYSCALE)
        if msk is None:
            raise ValueError(f"Cannot read mask: {msk_pth}")
        msk = cv2.resize(msk, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        # Binary mask 0/1
        msk = (msk > 0).astype(np.uint8)

        # Multi-class mask
        if self.classes:
            msk = np.isin(msk, self.class_ids).astype(np.uint8)

        return msk
    
    def __getitem__(self, idx):
        '''
            returns a Dictionary containing the sample data
        '''
        img_pth = self.image_paths[idx]
        msk_pth = self.mask_paths[idx] if self.mask_paths else None
        
        # --- cache ---
        if idx in self._cache:
            img = self._cache[idx]['img']
            msk = self._cache[idx].get('mask', None)
        else:
            img = self._read_image(img_pth)
            msk = self._read_mask(msk_pth) if msk_pth else None
            if len(self._cache) >= self._cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[idx] = {'img': img, 'mask': msk}
            
        
        # --- augmentation ---
        if msk is not None:
            aug = self.transforms(image=img, mask=msk)
            img_t, msk_t = aug['image'], aug['mask'].long()
        else:
            aug = self.transforms(image=img)
            img_t, msk_t = aug['image'], None
        
        # print(img_t.shape)
        # print(msk_t.shape)
        
        # ---- OUTPUT ----
        input = {'image': img_t}
        
        if msk_t is not None:
            input['mask'] = msk_t
        
        if self.requires_name:
            input['name'] = os.path.basename(img_pth)
            # print(input['name'].shape)
        
        input['source'] = 'Domain Based' if self.domain_based == True else "Other"
        return input
    
    def __len__(self): return len(self.image_paths) 
    
    
    
if __name__ == '__main__':
    config = Covid_CT_Scan_3_Config()
    all_labels, desired_labels = config.all_labels, config.only_labels
    trainImages, trainMasks = config.trainImagesDir, config.trainMasksDir
    
    dataset = UnetDataset(
        trainImages, isTrain=True, requires_name=True, masks=trainMasks, image_size=512,
        all_classes=all_labels, desired_classes=desired_labels
    )
    

    for i in range(len(dataset)):
        idx = np.random.randint(0, len(dataset) - 1)
        sample = dataset[i]
        img_t, msk_t, name = sample['image'], sample['mask'], sample['name']
        
        visualize_sample(img_t, msk_t, name) 
