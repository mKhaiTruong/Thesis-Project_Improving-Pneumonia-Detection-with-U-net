# Handle datasets
import os, sys, gc, random, time
import argparse
import numpy as np
import datetime

# Deep Learning
## Augmentation
from torch_optimizer import RAdam, Lookahead
from torch.optim.lr_scheduler import OneCycleLR
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import draw_bounding_boxes
from torch.optim import AdamW

# ignore warnings   
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# wandb
import wandb
# wandb.login()

# 0. Customization
sys.path.append(r"..\PneunomiaSeg\SAM\CUS SAM")
from dataset.concat_loaders import get_concat_loader
from train_val_test.train_one_epoch import train_one_epoch
from cus_segment_anything.build_sam import sam_model_registry
from utils.early_stopping import EarlyStopping
from utils.metrics import *
from utils.data_to_device import *
from sam_helpers.sam_helpers import *
from configuration.covid_ct import Covid_CT_Scan_Config
from configuration.covid_ct_2 import Covid_CT_Scan_2_Config
from configuration.covid_ct_3 import Covid_CT_Scan_3_Config
from configuration.covid_lesion import Covid_Lesion_Config
sam_ckpt_pth = r'..\PneunomiaSeg\SAM\CUS SAM\cus_segment_anything\sam_vit_b_01ec64.pth'

SEED = 42
isFP16Used = False 

# Set seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Optional: control hash seed for further determinism
os.environ["PYTHONHASHSEED"] = str(SEED)



def parse_args():
    work_dir = r'..\PneunomiaSeg\SAM\CUS SAM\results'
    parser = argparse.ArgumentParser(description='Medical Image - Pytorch Segmentation With SAM-Adapter')
    
    parser.add_argument('task', metavar='TASK', type=str, help='task id to use')
    parser.add_argument('--gpu_type', type=str, help='What is your GPU?')
    parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None, type=str,
                        help='path to the pretrained model')
    
    
    parser.add_argument("--work_dir", type=str, default=f'{work_dir}', help="work dir")
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
    parser.add_argument("--sam_checkpoint", type=str, default=sam_ckpt_pth)
    parser.add_argument("--metric_mode", type=str, default="min", choices=["max", "min"])
    parser.add_argument("--point_list", type=list, default=[1, 3, 5, 9], help="point_list")
    parser.add_argument("--multimask", type=bool, default=True, help="ouput multimask")
    
    parser.add_argument('--lr_scheduler', type=str, default='MultiStepLr', help='lr scheduler')
    parser.add_argument('--loss_type', type=str, default='focal_dice_loss', help='Loss Function')
    parser.add_argument("--use_amp", type=bool, default=True, help="use amp")
    
    args = parser.parse_args()
    return args



def main(args):
    global best_loss
    best_loss = 1e10
    
    # ---- CONFIGURATION ----
    args.lr          = 1e-4
    args.batch_size  = 8
    args.decay       = 5*1e-4
    args.patience    = 7
    args.start_epoch = 0
    args.epochs      = 200
    args.workers     = 0    # IF NO ERRORS THEN workers recommended > 0 for faster training
    args.seed        = 42
    args.print_freq  = 40
    args.iter_point  = 8
    args.mask_num    = 2
    args.image_size  = 256
    args.device      = 'cuda' if torch.cuda.is_available() else 'cpu'

    # <<< LOSS >>>
    args.bce_weight     = 0.3
    args.tversky_weight = 0.7
    args.pos_weight     = 4.0
    args.alpha  = 0.7
    args.beta   = 0.3
    args.smooth = 1.0
    
    RES_DIR = os.path.join(f'{args.work_dir}/models', args.task)
    os.makedirs(RES_DIR, exist_ok=True)
    os.makedirs(os.path.join(RES_DIR, 'logs'), exist_ok=True)
    save_hparams_to_json(args, os.path.join(RES_DIR, 'logs'))
    
    # >> MODEL
    model = sam_model_registry[args.model_type](args).to(args.device)
    
    # ---- OPTIMIZER (safe param grouping) ----
    # optimizer = Lookahead(base_optimizer, k=2, alpha=0.8)
    try:
        params = [
            {'params': model.decoder.parameters(),          'lr': args.lr},
            {'params': model.encoder.parameters(),          'lr': args.lr},
            {'params': model.segmentation_head.parameters(),'lr': args.lr},
        ]
        optimizer = AdamW(params, lr=args.lr, weight_decay=args.decay)
    except Exception:
        # fallback: use all parameters (works for any model)
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay)
    
    # Load pre-trained model
    if args.pre:
        if os.path.isfile(args.pre):
            
            print(f'=> Loading checkpoint {args.pre}')

            checkpoint = torch.load(args.pre, map_location=args.device)
            args.start_epoch = checkpoint['epoch']
            args.lr          = checkpoint['lr']
            best_loss        = checkpoint['best_loss']
            
            if args.start_epoch == args.epochs:
                args.epochs += args.epochs
            
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print(f"=> No check point found at {args.pre}")
    
    # >> LOSS FUNCTION
    criterions      = {
        'bcedice': BCEDiceLoss(
            bce_weight=args.bce_weight, dice_weight=1-args.bce_weight, 
            pos_weight=args.pos_weight, smooth=args.smooth
            ),
        'bcetversky': BCETverskyLoss(
            bce_weight=args.bce_weight, tversky_weight=1-args.bce_weight, 
            pos_weight=args.pos_weight, 
            alpha=args.alpha, beta=args.beta, smooth=args.smooth
            ),
        'focal': smp.losses.FocalLoss(mode='multilabel', alpha=0.25, gamma=2.0),
        'focal_dice_loss': FocalDiceloss_IoULoss()
    }
    early_stopper   = EarlyStopping(patience=args.patience, mode=args.metric_mode)

    if args.lr_scheduler:
        
        if args.lr_scheduler == 'OneCycleLR':
            scheduler = OneCycleLR(
                optimizer,
                max_lr = args.lr,
                steps_per_epoch = len(loaders['train']),
                epochs = args.epochs,
                pct_start = 2 / args.epochs  # warmup 2 epochs
            )
        
        elif args.lr_scheduler == 'MultiStepLr':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[5, 10], gamma=0.5
            )
    else:
        scheduler = None
        
    # >>> DATASET
    config_list = [
        Covid_CT_Scan_Config(),
        Covid_CT_Scan_2_Config()
    ] + [Covid_Lesion_Config()] * 2
    trainLoader = get_concat_loader(args, config_list)
    l = len(trainLoader)
    
    
    logger = get_logger(
        os.path.join(RES_DIR, 'logs', 
                     f'{args.task}_{datetime.datetime.now().strftime('%Y%m%d-%H%M.log')}')
    )
        

    for epoch in range(args.start_epoch, args.epochs):
        
        start = time.time()
        train_metrics = {}
        
        train_losses, train_iter_metrics = train_one_epoch(
            args, model, optimizer, trainLoader, epoch, criterions[args.loss_type]
        )
        
        if args.lr_scheduler is not None:
            scheduler.step()
        
        train_iter_metrics = [metric / l for metric in train_iter_metrics]
        train_metrics = {
            args.metrics[i]: '{:.4f}'.format(train_iter_metrics[i])
            for i in range(len(train_iter_metrics))
        }
        
        average_loss = np.mean(train_losses)
        lr = scheduler.get_last_lr()[0] if args.lr_scheduler is not None else args.lr
        end = time.time()

        if args.metric_mode == 'max':
            is_better = average_loss > best_loss
            best_loss = max(average_loss, best_loss)
        else:
            is_better = average_loss < best_loss
            best_loss = min(average_loss, best_loss)
        
        opt_state = optimizer.state_dict()
        save_checkpoint({
            'epoch'     : epoch+1,
            'arch'      : args.pre,
            'lr'        : lr,
            'state_dict': model.state_dict(),
            'best_loss' : best_loss,
            'optimizer' : opt_state
        }, is_better, args.task, RES_DIR)
        
        gpu_percent, cpu_percent = cpu_gpu_usage()
        logger.info(
            f'Epoch {epoch} took {(end - start)/60:.4f} minutes | '
                f'Train Loss = {average_loss:.4f} | '
                    f'GPU usage = {gpu_percent}% | '
                        f'CPU usage = {cpu_percent}% | '
                            f'Metrics = {train_metrics} | '
                                f'Average Loss = {average_loss:.4f} | '
                                    f'Best Loss = {best_loss:.4f}'
        )
        
        early_stopper(average_loss)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break
            
        gc.collect()
        torch.cuda.empty_cache()
            

if __name__ == '__main__':

    # ckp_pth = r'..\PneunomiaSeg\SAM\CUS SAM"\results\models\unet_deeplabv3plus_efficientnet-b3_exp2\unet_deeplabv3plus_efficientnet-b3_exp2_checkpoint.pth.tar'
    gpu_type = 'RTX3060'
    
    sys.argv = [
        'main.py',
        f'sam_adapter_exp1',
        '--gpu_type', f'{gpu_type}',
        # '--pre', ckp_pth
    ]
    
    '''
    exp1:
        Covid_CT_Scan_Config(),
        Covid_CT_Scan_2_Config()
        DomainBasedConfig: Covid_Lesion_Config()
    '''
    
    args = parse_args()
    main(args)
    