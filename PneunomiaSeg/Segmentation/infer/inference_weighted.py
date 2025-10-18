import os, re, glob, sys
import numpy as np
import torch
import cv2
from PIL import Image

sys.path.append(r"..\PneunomiaSeg\SAM\CUS SAM")
from models.unet_model import get_unet_model
from dataset.unet_dataset import Test_UnetDataset

# -----------------------------
# Config
# -----------------------------
compare_models = [
    ("deeplabv3plus", "efficientnet-b3"),
    ("deeplabv3plus", "mit_b1"),
    ("segformer",     "efficientnet-b3"),
    ("segformer",     "mit_b1"),
    ("unetpp",        "efficientnet-b3"),
]
exp_name = "exp1"

img_pth = [r"D:\Downloads\COVID-19 CT scans 3\ct_scans_png\scan_00098.png"]
msk_pth = r"D:\Downloads\COVID-19 CT scans 3\infected_masks_png\mask_00098.png"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
orig_img = Image.open(img_pth[0]).convert("RGB")
H, W     = np.array(orig_img).shape[:2]


# -----------------------------
# Helper: get best IOU from logs
# -----------------------------
def get_best_iou(model_type, encoder_type, exp_name="exp1"):
    log_dir = os.path.join(
        r"..\PneunomiaSeg\Segmentation\results\models",
        f"unet_{model_type}_{encoder_type}_{exp_name}",
        "logs"
    )
    best_iou = 0.0
    for f in glob.glob(os.path.join(log_dir, "*.log")):
        with open(f, "r", encoding="utf-8") as infile:
            for line in infile:
                m = re.search(r"Best IOU score = ([0-9.]+)", line)
                if m:
                    val = float(m.group(1))
                    best_iou = max(best_iou, val)
    return best_iou


# -----------------------------
# Helper: get probability map
# -----------------------------
def get_probability(model_type, encoder_type):
    ckpt_pth = os.path.join(
        r"..\PneunomiaSeg\Segmentation\results\models",
        f"unet_{model_type}_{encoder_type}_{exp_name}",
        f"unet_{model_type}_{encoder_type}_{exp_name}_model_best.pth.tar"
    )
    test_ds = Test_UnetDataset(img_pth)
    img = test_ds[0]['image'].unsqueeze(0).to(device)
    checkpoint = torch.load(ckpt_pth, map_location=device, weights_only=False)
    model = get_unet_model(model_type, encoder_type).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    with torch.no_grad():
        pred = model(img)
        prob = torch.sigmoid(pred).squeeze().cpu().numpy()
    prob = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)
    return prob


# -----------------------------
# Ensemble (Weighted soft voting)
# -----------------------------
probs, weights = [], []
for model_type, encoder_type in compare_models:
    prob = get_probability(model_type, encoder_type)
    weight = get_best_iou(model_type, encoder_type, exp_name)
    probs.append(prob * weight)
    weights.append(weight)

ensemble_prob = np.sum(probs, axis=0) / np.sum(weights)
ensemble_mask = (ensemble_prob > 0.5).astype(np.uint8)


# -----------------------------
# Visualization
# -----------------------------
import matplotlib.pyplot as plt
gt_msk = np.array(Image.open(msk_pth).convert("L"))
gt_msk = (gt_msk > 127).astype(np.uint8)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.imshow(orig_img); plt.title("Original"); plt.axis("off")
plt.subplot(1,3,2); plt.imshow(gt_msk, cmap="gray"); plt.title("Ground Truth"); plt.axis("off")
plt.subplot(1,3,3); plt.imshow(ensemble_mask, cmap="gray"); plt.title("Weighted Ensemble"); plt.axis("off")
plt.tight_layout(); plt.show()
