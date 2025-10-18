import sys
import numpy as np
import torch



import torch, sys, numpy as np
sys.path.append(r"D:\Deep_Learning_Object_Detection\sam2")

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def init_sam2_predictor(cfg_path, checkpoint_path, device='cuda'):
    sam = build_sam2(cfg_path, checkpoint_path)
    sam.to(device)
    predictor = SAM2ImagePredictor(sam)
    return predictor

def sam2_predict_with_points(predictor, img, points, labels, multimask = True):
    
    point_coords = np.array(points, dtype=np.float32)
    point_labels = np.array(labels, dtype=np.int64)
    
    predictor.set_image(img)
    with torch.inference_mode():
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else torch.cpu.amp.autocast(enabled=False)
        with ctx:
            masks, scores, logits = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=multimask,
            )

    best_idx = int(np.argmax(scores))
    best_mask = masks[best_idx].astype(np.uint8)
    best_score = float(scores[best_idx])

    return best_mask, best_score, logits[best_idx]

def sam2_predict_with_box(predictor, img, mask, multimask=False):

    # Get bbox from binary mask
    y, x = np.where(mask > 0)
    if len(x) == 0 or len(y) == 0:
        print(f"[WARN] No object region found in mask for {img_pth}")
        return None, None, None

    box = np.array([[x.min(), y.min(), x.max(), y.max()]], dtype=np.float32)

    predictor.set_image(img)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        masks, scores, logits = predictor.predict(
            box=box,
            multimask_output=multimask,
        )

    best_idx = np.argmax(scores)
    masks = masks.astype(np.uint8)
    return masks[best_idx], scores[best_idx], logits[best_idx]


def sam2_refine_mask(predictor, img, mask, multimask=False):
    
    if img.ndim == 3 and img.shape[0] == 3:  # (3, 256, 256)
        img = img.transpose(1, 2, 0)
    mask = (mask > 0).astype(np.float32)

    # (H, W) -> (1, 1, H, W)
    if mask.ndim == 2:
        mask = mask[None, None, ...]
    elif mask.ndim == 3:
        mask = mask[None, ...]
    mask = torch.from_numpy(mask).to(predictor.device)
    
    predictor.set_image(img)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=None,
            mask_input=mask,
            multimask_output=False
        )

    best_idx = np.argmax(scores)
    masks = masks.astype(np.uint8)
    return masks[best_idx], scores[best_idx], logits[best_idx]
