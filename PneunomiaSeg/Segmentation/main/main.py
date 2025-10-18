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

from utils.early_stopping import EarlyStopping
from utils.loss_fncs import *
from utils.data_to_device import *
from configuration.covid_ct import Covid_CT_Scan_Config
from configuration.covid_ct_2 import Covid_CT_Scan_2_Config
from configuration.covid_lesion import Covid_Lesion_Config


from dataset.concat_loader import get_concat_loader
from models.unet_model import get_unet_model
from train_val_test.train_one_epoch import train_one_epoch
from train_val_test.validate import cus_valid



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
    work_dir = r'..\PneunomiaSeg\Segmentation\results'
    parser = argparse.ArgumentParser(description='Medical Image - Pytorch Segmentation')
    
    parser.add_argument('task', metavar='TASK', type=str, help='task id to use')
    parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None, type=str,
                        help='path to the pretrained model')
    parser.add_argument("--work_dir", type=str, default=f'{work_dir}', help="work dir")
    parser.add_argument("--metrics", nargs='+', default=['bcedice', 'bcetversky', 'focal'], help="metrics")
    parser.add_argument("--model_type", type=str, default=None, help="unet model_type")
    parser.add_argument("--encoder_type", type=str, default=None, help="unet encoder_type")
    parser.add_argument('--lr_scheduler', type=str, default='OneCycleLR', help='lr scheduler')
    parser.add_argument('--loss_type', type=str, default='bcetversky', help='loss function')
    parser.add_argument("--metric_mode", type=str, default="max", choices=["max", "min"])
    parser.add_argument("--point_list", type=list, default=[1, 3, 5, 9], help="point_list")
    parser.add_argument("--multimask", type=bool, default=True, help="ouput multimask")
    parser.add_argument("--use_amp", type=bool, default=True, help="use amp")
    
    args = parser.parse_args()
    return args



def main(args):
    global best_iou
    best_iou = 0.0
    
    # ---- CONFIGURATION ----
    args.lr          = 1e-4
    args.batch_size  = 8
    args.decay       = 5*1e-4
    args.patience    = 7
    args.start_epoch = 0
    args.epochs      = 200
    args.workers     = 12
    args.seed        = 42
    args.print_freq  = 40
    args.image_size  = 256
    args.device      = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # <<< LOSS >>>
    args.loss_type      = 'bcetversky'
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
    model = get_unet_model(args.model_type, args.encoder_type).to(args.device)
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
            best_iou         = checkpoint['best_iou']
            
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
        'focal': smp.losses.FocalLoss(mode='multilabel', alpha=0.25, gamma=2.0)
    }
    early_stopper   = EarlyStopping(patience=args.patience, mode=args.metric_mode)

    
    # >>> DATASET
    config_list = [
        Covid_CT_Scan_Config(),
        Covid_CT_Scan_2_Config()
    ] + [Covid_Lesion_Config()] * 2
    loaders = get_concat_loader(args, config_list)
    
    if args.lr_scheduler:
        scheduler = OneCycleLR(
            optimizer,
            max_lr = args.lr,
            steps_per_epoch = len(loaders['train']),
            epochs = args.epochs,
            pct_start = 2 / args.epochs  # warmup 2 epochs
        )
    else:
        scheduler = None
    
    
    logger = get_logger(
        os.path.join(RES_DIR, 'logs', 
                     f'{args.task}_{datetime.datetime.now().strftime('%Y%m%d-%H%M.log')}')
    )
        
    
            

    for epoch in range(args.start_epoch, args.epochs):
        start = time.time()
        avg_train_loss  = train_one_epoch(
            model, loaders['train'], criterions[args.loss_type], optimizer, scheduler, epoch, args
        )
        
        if epoch % 2 == 0:
            avg_val_loss, mean_iou = cus_valid(
                model, loaders['valid'], criterions[args.loss_type], epoch, args
            )
            if args.lr_scheduler is not None:
                scheduler.step()
                
            if args.metric_mode == 'max':
                is_better = mean_iou > best_iou
                best_iou   = max(mean_iou, best_iou)
            else:
                is_better = mean_iou < best_iou
                best_iou   = min(mean_iou, best_iou)
            
            lr = scheduler.get_last_lr()[0] if args.lr_scheduler is not None else args.lr
            opt_state = optimizer.state_dict()
            save_checkpoint({
                'epoch'     : epoch+1,
                'lr'        : lr,
                'arch'      : args.pre,
                'state_dict': model.state_dict(),
                'best_iou'  : best_iou,
                'optimizer' : opt_state
            }, is_better, args.task, RES_DIR)

            end = time.time()
            logger.info(
                f'Epoch {epoch} took {(end - start)/60:.4f} minutes | '
                    f'Train Loss = {avg_train_loss:.4f} | '
                        f'Valid Loss = {avg_val_loss:.4f} | '
                            f'Current IOU score = {mean_iou:.4f} | '
                                f'Best IOU score = {best_iou:.4f}'
            )
        
            score = -0.25*avg_train_loss -0.25*avg_val_loss + 0.5*mean_iou
            early_stopper(score)
            if early_stopper.early_stop:
                print("Early stopping triggered.")
                break
            
            gc.collect()
            torch.cuda.empty_cache()
            

if __name__ == '__main__':

    # ckp_pth = r'..\PneunomiaSeg\Segmentation\results\models\unet_unetpp_efficientnet-b3_exp1\unet_unetpp_efficientnet-b3_exp1_checkpoint.pth.tar'
    model_type = 'unetpp'
    model_type = model_type.lower()
    encoder_type = 'efficientnet-b3'        # efficientnet-b3, mit_b1
    
    sys.argv = [
        'main.py',
        f'unet_{model_type}_{encoder_type}_exp1',
        "--model_type", model_type,
        "--encoder_type", encoder_type,
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
    