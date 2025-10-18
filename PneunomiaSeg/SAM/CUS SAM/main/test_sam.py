import sys, os
import argparse
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.append(r"..\PneunomiaSeg\SAM\CUS SAM")
sam_ckpt_pth = r'..\PneunomiaSeg\SAM\CUS SAM\cus_segment_anything\sam_vit_b_01ec64.pth'
from utils.metrics import *
from sam_helpers.sam_helpers import *
from cus_segment_anything.build_sam import sam_model_registry
from dataset.sam_adapter_dataset import TestingDataset
from configuration.covid_ct_3 import Covid_CT_Scan_3_Config

# -----------------------------
# Config
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Medical Image - Pytorch Segmentation With SAM-Adapter')
    
    parser.add_argument('task', metavar='TASK', type=str, help='task id to use')
    parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None, type=str,
                        help='path to the pretrained model')
    
    parser.add_argument("--boxes_prompt", type=bool, default=True, help="use boxes prompt")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--sam_checkpoint", type=str, default=sam_ckpt_pth)
    
    parser.add_argument("--point_num", type=int, default=1, help="point num")
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
    parser.add_argument("--metric_mode", type=str, default="min", choices=["max", "min"])
    parser.add_argument("--point_list", type=list, default=[1, 3, 5, 9], help="point_list")
    parser.add_argument("--multimask", type=bool, default=True, help="ouput multimask")
    parser.add_argument("--iter_point", type=int, default=1, help="iter num") 
    parser.add_argument('--lr_scheduler', type=str, default='MultiStepLr', help='lr scheduler')
    parser.add_argument('--loss_type', type=str, default='focal_dice_loss', help='Loss Function')
    parser.add_argument("--use_amp", type=bool, default=True, help="use amp")
    
    args = parser.parse_args()
    if args.iter_point > 1:
        args.point_num = 1
    return args



def main(args, test_loader):
    
    # -----------------------------
    # Init SAM Predictor
    # -----------------------------
    # >>> SAM ADAPTER
    sama = sam_model_registry[args.model_type](args).to(args.device) 
    checkpoint = torch.load(args.pre, weights_only=False, map_location=args.device)
    sama.load_state_dict(checkpoint['state_dict'])
    
    # -----------------------------
    # TEST FUNCTIONS
    # -----------------------------
    criterion = FocalDiceloss_IoULoss()
    test_pog_bar = tqdm(test_loader)
    l = len(test_loader)   
    
    sama.eval()
    test_loss = []
    test_iter_metrics = [0] * len(args.metrics)
    test_metrics = {}
    
    for i, batched_input in enumerate(test_pog_bar):
        batched_input = to_device(batched_input, args.device) 
        
        img, msk = batched_input["image"], batched_input["label"]
        ori_mask = batched_input["ori_label"]
        original_size = batched_input["original_size"]
        img_name = batched_input['name'][0]
        
        with torch.no_grad():
            image_embeddings = sama.image_encoder(img)
        
        if args.boxes_prompt:
            batched_input["point_coords"], batched_input["point_labels"] = None, None
            masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, sama, image_embeddings)
        
        masks, pad = postprocess_masks(low_res_masks, args.image_size, original_size)
        loss = criterion(masks, ori_mask, iou_predictions)
        test_loss.append(loss.item())
        
        test_batch_metrics = SegMetrics(masks, ori_mask, args.metrics)
        test_batch_metrics = [float('{:.4f}'.format(metric)) for metric in test_batch_metrics]
        
        for j in range(len(args.metrics)):
            test_iter_metrics[j] += test_batch_metrics[j]
    
    test_iter_metrics = [metric / l for metric in test_iter_metrics]
    test_metrics = {args.metrics[i]: '{:.4f}'.format(test_iter_metrics[i]) for i in range(len(test_iter_metrics))}
    
    average_loss = np.mean(test_loss)
    print(f"Test loss: {average_loss:.4f}, metrics: {test_metrics}")


if __name__ == '__main__':
    
    # -----------------------------
    # Config
    # -----------------------------
    config = Covid_CT_Scan_3_Config()
    all_labels, desired_labels = config.all_labels, config.only_labels
    train_img_pth = config.trainImagesDir
    train_msk_pth = config.trainMasksDir
    test_dataset = TestingDataset(
        train_img_pth, train_msk_pth, requires_name=True, 
        return_og_msk=True, all_classes=all_labels, desired_classes=desired_labels, 
        image_size=256, point_num=1
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    model_dir = r"..\PneunomiaSeg\SAM\CUS SAM\results\models"
    exp_name  = "exp1"
    fine_tune = os.path.join(
        model_dir, 
        f'sam_adapter_{exp_name}',
        f'sam_adapter_{exp_name}_model_best.pth.tar'
    )

    sys.argv = [
        'test_sam.py',
        f'Test SAM',
        '--pre', fine_tune
    ]
    
    args = parse_args()
    main(args, test_loader)