import sys
from tqdm import tqdm
import torch
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(r"..\PneunomiaSeg\SAM\CUS SAM")
from inferent.inference_helpers import *
from inferent.sam_adapter_helpers import *
test_loader = get_test_loader()

def denormalize(img):
    pixels_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    pixels_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    pixels_mean = pixels_mean[:, np.newaxis, np.newaxis]
    pixels_std = pixels_std[:, np.newaxis, np.newaxis]
    img = img * pixels_std + pixels_mean
    return img
def visualize_results(img, ground_truth_mask, sam2_mask, sama_mask, img_name, original_size):
    img = denormalize(img)
    
    # Ensure img is in HWC format before resizing
    if img.ndim == 3 and img.shape[0] == 3:  # CHW
        img = img.transpose(1, 2, 0)
    
    img_resized = cv2.resize(img, (original_size[1], original_size[0]))
    ground_truth_mask_resized = cv2.resize(ground_truth_mask, (original_size[1], original_size[0]))
    sam2_mask_resized = cv2.resize(sam2_mask, (original_size[1], original_size[0]))

    if isinstance(sama_mask, torch.Tensor):
        if len(sama_mask.shape) == 4:  # [1,1,H,W]
            sama_mask_np = sama_mask.squeeze().detach().cpu().numpy()
        elif len(sama_mask.shape) == 3:  # [1,H,W]
            sama_mask_np = sama_mask.squeeze().detach().cpu().numpy()
        else:
            sama_mask_np = sama_mask.detach().cpu().numpy()
    else:
        sama_mask_np = sama_mask

    binary_sam2_mask = (sam2_mask_resized > 0.8).astype(np.uint8)
    binary_sama_mask = (sama_mask_np > 0.5).astype(np.uint8)
    
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_resized.astype(np.uint8))
    axes[0].imshow(ground_truth_mask_resized, alpha=0.5, cmap='jet')
    axes[0].set_title(f"Original + Ground Truth Mask: {img_name}")
    axes[0].axis('off')

    axes[1].imshow(img_resized.astype(np.uint8))
    axes[1].imshow(binary_sam2_mask, alpha=0.5)
    axes[1].set_title("Original + SAM2 Predicted Mask")
    axes[1].axis('off')

    axes[2].imshow(img_resized.astype(np.uint8))
    axes[2].imshow(binary_sama_mask, alpha=0.5)
    axes[2].set_title("Original + SAM-Adapter Predicted Mask")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()




def main(args):
    # -----------------------------
    # Init SAM Predictor
    # -----------------------------
    # >>> BASE SAM2
    sam  = init_sam2_predictor(
        cfg_path=r"..\Segmentation\sam\sam2.1_hiera_b+.yaml",
        checkpoint_path=r"..\Segmentation\sam\sam2.1_hiera_base_plus.pt",
        device='cuda'
    )
    
    # >>> MY SAM-Adapter
    sama = init_sam_adapter_predictor(args)
    
    # ----------------
    # SAM VS
    # ----------------
    test_pog_bar = tqdm(test_loader)
    sama.eval()
    for i, batched_input in enumerate(test_pog_bar):
        batched_input = to_device(batched_input, args.device) 
        
        img, msk = batched_input["image"], batched_input["label"]
        ori_mask = batched_input["ori_label"]
        original_size = batched_input["original_size"]
        img_name = batched_input['name'][0]
        
        # print(f'img = {img.shape}')
        # print(f'msk = {msk.shape}')
        # print(f'ori_mask = {ori_mask.shape}')
        
        # -----------------
        # OUTPUT 
        # -----------------
        # >>> BASE SAM2
        img_np = img.squeeze().cpu().numpy()
        msk_np = msk.squeeze().cpu().numpy()
        # print(f'img_np = {img_np.shape}')    
        # print(f'msk_np = {msk_np.shape}')
        sam_mask, _, _ = sam2_refine_mask(sam, img_np, msk_np)
        
        # >>> MY SAM-ADAPTER
        sama_mask = sam_adapter_refine_mask(args, sama, batched_input, original_size)

        # print(f'sam_mask = {sam_mask.shape}')    
        # print(f'sama_mask = {sama_mask.shape}')  
        # print(f'ori_mask = {ori_mask.shape}')  
        h, w = ori_mask.shape[2], ori_mask.shape[3]
        visualize_results(img_np, msk_np, sam_mask, sama_mask, img_name, (h, w))
    
if __name__ == '__main__':
    
    
    model_dir = r"D:\Deep_Learning_Object_Detection\randProjects\PneunomiaSeg\SAM\CUS SAM\results\models"
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
    main(args)
