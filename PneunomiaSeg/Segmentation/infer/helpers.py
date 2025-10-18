import os, sys
import numpy as np
import torch
import cv2

sys.path.append(r"..\PneunomiaSeg\SAM\CUS SAM")
from dataset.unet_dataset import Test_UnetDataset
from models.unet_model import get_unet_model

# -----------------------------
# Predict mask
# -----------------------------
def get_prediction(model_dir, model_type, encoder_type, exp_name, img, device='cuda'):
    
    W, H = img.shape[2], img.shape[3]
    
    ckpt_pth = os.path.join(
        model_dir,
        f"unet_{model_type}_{encoder_type}_{exp_name}",
        f"unet_{model_type}_{encoder_type}_{exp_name}_model_best.pth.tar"
    )
    
    checkpoint = torch.load(ckpt_pth, map_location=device, weights_only=False)
    model = get_unet_model(model_type, encoder_type).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    with torch.no_grad():
        pred = model(img)
        prob = torch.sigmoid(pred)
        pred_msk = (prob > 0.5).float().squeeze().cpu().numpy()
    pred_msk = cv2.resize(pred_msk.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    return pred_msk


# -----------------------------
# From predict mask to central points
# -----------------------------
def mask_to_components(mask: np.ndarray, min_area: int = 10):
    if mask is None:    return []
    bin_mask = (mask > 0).astype(np.uint8) * 255
    
    # External contours
    contours, _ = cv2.findContours(bin_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    comps = []
    cid = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
            
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            pts = cnt.reshape(-1, 2)
            cy, cx = int(np.mean(pts[:,1])), int(np.mean(pts[:,0]))
        
        x, y, w, h = cv2.boundingRect(cnt)
        bbox = (x, y, x + w - 1, y + h - 1)

        comps.append({
            "id": cid,
            "centroid": (cx, cy),
            "area": float(area),
            "bbox": bbox,
            "contour": cnt
        })
        cid += 1
    
    comps = sorted(comps, key=lambda d: d["area"], reverse=True)
    return comps

def mask_to_centers(mask: np.ndarray, min_area:int=50):
    comps = mask_to_components(mask, min_area=min_area)
    return [c["centroid"] for c in comps]