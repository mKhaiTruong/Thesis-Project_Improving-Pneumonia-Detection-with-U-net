from torch.utils.data import ConcatDataset, DataLoader
from sklearn.model_selection import train_test_split
import sys, torch

sys.path.append(r"..\PneunomiaSeg\SAM\CUS SAM")
from dataset.unet_dataset import UnetDataset

# For SAM
# def collate_fn(batch):
#     images = torch.stack([x['image'] for x in batch])
#     masks  = torch.stack([x['mask'] for x in batch])
#     sources = [x.get('source', 'Other') for x in batch]
#     return {'image': images, 'mask': masks, 'source': sources}

def get_loaders(trainDataset, validDataset, args):
    nw = max(0, min(int(args.workers), 2))
    
    trainLoader = DataLoader(
        trainDataset,
        batch_size=args.batch_size,
        shuffle=True,                
        num_workers=nw,
        pin_memory=True,
        persistent_workers=(nw > 0),
        prefetch_factor=(2 if nw > 0 else None),
        drop_last=True,
        # collate_fn=collate_fn,      
    )
    validLoader = DataLoader(
        validDataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=(nw > 0),
        prefetch_factor=(2 if nw > 0 else None),
        drop_last=False,
        # collate_fn=collate_fn,  
    )
    return {'train': trainLoader, 'valid': validLoader}

def get_cus_dataset(args, config, test_size=0.2):
    
    all_labels, desired_labels = config.all_labels, config.only_labels
    domain_based = True if config.domain_based else False
    
    if hasattr(config, "validImagesDir") and hasattr(config, "validMasksDir") \
       and len(config.validImagesDir) > 0 and len(config.validMasksDir) > 0:
        
        # print(">> Using predefined validation set from config")
        train_img, train_msk = config.trainImagesDir, config.trainMasksDir
        valid_img, valid_msk = config.validImagesDir, config.validMasksDir

    else:
        # print(">> Splitting train into train/valid")
        train_img, valid_img, train_msk, valid_msk = train_test_split(
            config.trainImagesDir, config.trainMasksDir,
            test_size=test_size, random_state=42
        )
    
    train_dataset = UnetDataset(
        train_img, isTrain=True, masks=train_msk,  
        image_size=args.image_size, 
        all_classes=all_labels, desired_classes=desired_labels, domain_based=domain_based
    )
    valid_dataset = UnetDataset(
        valid_img, isTrain=False, masks=valid_msk, 
        image_size=args.image_size, 
        all_classes=all_labels, desired_classes=desired_labels, domain_based=domain_based
    )
    
    return train_dataset, valid_dataset

def get_concat_loader(args, config_list, test_size=0.2):
    train_datasets = []
    valid_datasets = []
    
    for cfg in config_list:
        train_ds, valid_ds = get_cus_dataset(args, cfg, test_size=test_size)
        train_datasets.append(train_ds)
        valid_datasets.append(valid_ds)
    
    combined_train_dataset = ConcatDataset(train_datasets)
    combined_valid_dataset = ConcatDataset(valid_datasets)
    
    loaders = get_loaders(combined_train_dataset, combined_valid_dataset, args)
    return loaders

