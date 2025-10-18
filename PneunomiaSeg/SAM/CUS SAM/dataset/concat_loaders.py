import sys
from torch.utils.data import ConcatDataset, DataLoader

sys.path.append(r"..\PneunomiaSeg\SAM\CUS SAM")
from dataset.sam_adapter_dataset import TrainingDataset

def collate_fn(batch): return tuple(zip(*batch))

def get_loaders(trainDataset, args):
    nw = max(0, int(args.workers))
    
    trainLoader = DataLoader(
        trainDataset,
        batch_size=args.batch_size,
        shuffle=True,                
        num_workers=nw,
        pin_memory=True,
        persistent_workers=(nw > 0),
        prefetch_factor=(4 if nw > 0 else None),
        drop_last=True,
        # collate_fn=collate_fn,
    )
    return trainLoader


def get_cus_dataset(args, config):
    
    all_labels, desired_labels = config.all_labels, config.only_labels
    train_img_pth = config.trainImagesDir
    train_msk_pth = config.trainMasksDir
    
    train_dataset = TrainingDataset(
        train_img_pth, train_msk_pth, requires_name=False,
        all_classes=all_labels, desired_classes=desired_labels,
        image_size=args.image_size, point_num=1, mask_num=args.mask_num
    )
    
    return train_dataset

def get_concat_loader(args, config_list):
    
    train_datasets = []
    for cfg in config_list:
        train_ds = get_cus_dataset(args, cfg)
        train_datasets.append(train_ds)

    
    combined_train_dataset = ConcatDataset(train_datasets)
    
    trainLoader = get_loaders(combined_train_dataset, args)
    return trainLoader


