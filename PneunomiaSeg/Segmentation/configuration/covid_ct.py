import os, sys, cv2, numpy as np
from glob import glob

sys.path.append(r"..\PneunomiaSeg\SAM\CUS SAM")
from configuration.base_config import BaseConfig
from utils.visualization import visualize_jpg_exif

class Covid_CT_Scan_Config(BaseConfig):
    def __init__(self):
        super().__init__()
        
        self.all_labels  = ['background', 'pneu']
        self.only_labels = ['pneu']
        self.parentDir = r"D:\Downloads\COVID-19 CT scans"
        
        # >> TRAIN
        self.trainImagesDir = sorted(glob(
            pathname   = os.path.join(self.parentDir, 'ct_scans_png', '*.png'), 
            recursive  = True
        ))    
        self.trainMasksDir = sorted(glob(
            pathname   = os.path.join(self.parentDir, 'infected_masks_png', '*.png'), 
            recursive  = True
        ))


if __name__ == '__main__':
    
    config = Covid_CT_Scan_Config()
    img_pths, msk_pths = config.trainImagesDir, config.trainMasksDir
    # test_images = config.testDir
    
    print("Num images:", len(img_pths))
    print("Num masks :", len(msk_pths))
    visualize_jpg_exif(img_pths, msk_pths)