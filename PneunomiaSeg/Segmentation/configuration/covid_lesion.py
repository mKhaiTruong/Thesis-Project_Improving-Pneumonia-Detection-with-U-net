import os, sys, cv2, numpy as np
from glob import glob

sys.path.append(r"..\PneunomiaSeg\Segmentation")
from configuration.base_config import BaseConfig
from utils.visualization import visualize_jpg_exif

class Covid_Lesion_Config(BaseConfig):
    def __init__(self):
        super().__init__()
        
        self.all_labels  = ['background', 'pneu']
        self.only_labels = ['pneu']
        self.parentDir = r"D:\Downloads\COVID-19_lesion_seg\subset_step4_skip100"
        
        # >> TRAIN
        self.trainImagesDir = sorted(glob(
            pathname   = os.path.join(self.parentDir, 'frames', '*.png'), 
            recursive  = True
        ))    
        self.trainMasksDir = sorted(glob(
            pathname   = os.path.join(self.parentDir, 'masks', '*.png'), 
            recursive  = True
        ))


if __name__ == '__main__':
    
    config = Covid_Lesion_Config()
    img_pths, msk_pths = config.trainImagesDir, config.trainMasksDir
    # test_images = config.testDir
    
    print("Num images:", len(img_pths))
    print("Num masks :", len(msk_pths))
    visualize_jpg_exif(img_pths, msk_pths)