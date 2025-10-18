# Handle datasets
import os, sys, random, cv2
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Deep Learning
## Augmentation
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


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

from utils.loss_fncs import *
from utils.data_to_device import *
from configuration.covid_ct_3 import Covid_CT_Scan_3_Config
from dataset.unet_dataset import Test_UnetDataset
from dataset.aug import reverse_normalize_tensor
from models.unet_model import get_unet_model


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

SAV_DIR = r'D:\Downloads\COVID-19 CT scans 3\unet_unetpp_b3_22epochs_masks'
os.makedirs(SAV_DIR, exist_ok=True)

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
    
    # ---- CONFIGURATION ----
    args.lr          = 1e-4
    args.batch_size  = 1
    args.decay       = 5*1e-4
    args.patience    = 7
    args.start_epoch = 0
    args.epochs      = 200
    args.workers     = 12
    args.seed        = 42
    args.print_freq  = 40
    args.image_size  = 256
    args.device      = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # >> MODEL
    model = get_unet_model(args.model_type, args.encoder_type).to(args.device)
    
    # Load pre-trained model
    if args.pre:
        if os.path.isfile(args.pre):
            
            print(f'=> Loading checkpoint {args.pre}')
            checkpoint = torch.load(args.pre, map_location=args.device)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print(f"=> No check point found at {args.pre}")
    
    # >>> DATASET
    config  = Covid_CT_Scan_3_Config()
    test_ds = Test_UnetDataset(config.trainImagesDir, img_size=args.image_size) 
    loader  = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False,
    )
    
    prog_bar = tqdm(loader, total=len(loader))
    model.eval()
    for i, batch in enumerate(prog_bar):
        img = batch['image'].to(args.device)
        
        with torch.no_grad():
            out = model(img)
            msk = torch.sigmoid(out)
            
            # POST_PROCESS PREDICTED MASKS
            mask_np = msk.squeeze().detach().cpu().numpy()      # [H,W]
            mask_bin = (mask_np > 0.9).astype(np.uint8)
            name = batch['name'][0]
            
            # SAVING PREDICTIONS
            save_path = os.path.join(SAV_DIR, name)
            cv2.imwrite(save_path, mask_bin * 255)
            
            # -----------------------------------------------------------------------
            # SANITY CHECKING
            # img_vis = reverse_normalize_tensor(img)             # [H,W,C], [0,1]
            # plt.figure(figsize=(8,4))
            
            # plt.subplot(1,2,1)
            # plt.imshow(img_vis)
            # plt.title(f'Input Image: {name}')
            # plt.axis('off')

            # plt.subplot(1,2,2)
            # plt.imshow(mask_bin)
            # plt.title('Predicted Mask')
            # plt.axis('off')

            # plt.show()

            # if i == 10:
            #     break
            # ------------------------------------------------------------------
        

if __name__ == '__main__':

    ckp_pth = r'..\PneunomiaSeg\Segmentation\results\models\unet_unetpp_efficientnet-b3_exp1\unet_unetpp_efficientnet-b3_exp1_model_best.pth.tar'
    model_type = 'unetpp'
    model_type = model_type.lower()
    encoder_type = 'efficientnet-b3'
    
    sys.argv = [
        'main.py',
        f'Testing U-net Model',
        "--model_type", model_type,
        "--encoder_type", encoder_type,
        '--pre', ckp_pth
    ]
    
    args = parse_args()
    main(args)
    