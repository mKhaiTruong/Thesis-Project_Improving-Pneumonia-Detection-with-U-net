import sys
from tqdm import tqdm
import torch
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(r"..\PneunomiaSeg\SAM\CUS SAM")
from inferent.inference_helpers import *
from inferent.sam_adapter_helpers import *
test_loader = get_test_loader_for_refinement()

SAVE_DIR = r'D:\Downloads\COVID-19 CT scans 3\sam_adapter_masks'
os.makedirs(SAVE_DIR, exist_ok=True)

def denormalize(img):
    pixels_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    pixels_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    pixels_mean = pixels_mean[:, np.newaxis, np.newaxis]
    pixels_std = pixels_std[:, np.newaxis, np.newaxis]
    img = img * pixels_std + pixels_mean
    return img

def visualize_results(img, ground_truth_mask, binary_mask, img_name, original_size):
    img = denormalize(img)
    
    # Ensure img is in HWC format before resizing
    if img.ndim == 3 and img.shape[0] == 3:  # CHW
        img = img.transpose(1, 2, 0)
    
    img_resized = cv2.resize(img, (original_size[1], original_size[0]))
    ground_truth_mask_resized = cv2.resize(ground_truth_mask, (original_size[1], original_size[0]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(img_resized.astype(np.uint8))
    axes[0].imshow(ground_truth_mask_resized, alpha=0.5, cmap='jet')
    axes[0].set_title(f"Original + Ground Truth Mask: {img_name}")
    axes[0].axis('off')

    axes[1].imshow(img_resized.astype(np.uint8))
    axes[1].imshow(binary_mask, alpha=0.5)
    axes[1].set_title("Original + SAM-Adapter Predicted Mask")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()




def main(args):
    # -----------------------------
    # Init SAM Predictor
    # -----------------------------
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
        
        # -----------------
        # OUTPUT 
        # -----------------
        img_np = img.squeeze().cpu().numpy()
        msk_np = msk.squeeze().cpu().numpy()
        
        # >>> MY SAM-ADAPTER
        sama_mask = sam_adapter_refine_mask(args, sama, batched_input, original_size)
        if isinstance(sama_mask, torch.Tensor):
            if len(sama_mask.shape) == 4:  # [1,1,H,W]
                sama_mask_np = sama_mask.squeeze().detach().cpu().numpy()
            elif len(sama_mask.shape) == 3:  # [1,H,W]
                sama_mask_np = sama_mask.squeeze().detach().cpu().numpy()
            else:
                sama_mask_np = sama_mask.detach().cpu().numpy()
        else:
            sama_mask_np = sama_mask
            
        binary_mask = (sama_mask_np > 0.7).astype(np.uint8)
        h, w = ori_mask.shape[2], ori_mask.shape[3]
        binary_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
        # ----- Save mask -----
        save_path = os.path.join(SAVE_DIR, img_name)
        cv2.imwrite(save_path, binary_mask*255)
    
        # SANITY CHECK
        # print(f'sama_mask = {sama_mask.shape}')  
        # print(f'ori_mask = {ori_mask.shape}')  
        # visualize_results(img_np, msk_np, binary_mask, img_name, (h, w))
    
if __name__ == '__main__':
    
    
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
    main(args)
