import os, sys, argparse
import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(r"..\PneunomiaSeg\SAM\CUS SAM")
from dataset.unet_dataset import Test_UnetDataset
from models.unet_model import get_unet_model


# -----------------------------
# Config
# -----------------------------
exp_name = 'exp1'
compare_models = [
    ("deeplabv3plus", "efficientnet-b3"),
    ("deeplabv3plus", "mit_b1"),
    ("segformer",     "efficientnet-b3"),
    ("segformer",     "mit_b1"),
    ("unetpp",        "efficientnet-b3"),
]

img_pth = [r"D:\Downloads\COVID-19 CT scans 3\ct_scans_png\scan_00502.png"]
msk_pth = r"D:\Downloads\COVID-19 CT scans 3\infected_masks_png\mask_00502.png"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
orig_img = Image.open(img_pth[0]).convert("RGB")
gt_msk   = np.array(Image.open(msk_pth).convert("L"))
gt_msk   = (gt_msk > 127).astype(np.uint8)
H, W     = np.array(orig_img).shape[:2]


# -----------------------------
# Helper to predict PROB map
# -----------------------------
def get_prob_map(model_type, encoder_type):
    ckpt_pth = os.path.join(
        r"..\PneunomiaSeg\Segmentation\results\models",
        f"unet_{model_type}_{encoder_type}_{exp_name}",
        f"unet_{model_type}_{encoder_type}_{exp_name}_model_best.pth.tar"
    )
    img = Test_UnetDataset(img_pth)[0]['image'].unsqueeze(0).to(device)
    checkpoint = torch.load(ckpt_pth, map_location=device, weights_only=False)
    model = get_unet_model(model_type, encoder_type).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    with torch.no_grad():
        pred = model(img)              # logits
        prob = torch.sigmoid(pred)     # (1,1,H,W)
        prob = prob.squeeze().cpu().numpy()
    prob = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)  # giữ soft prob
    return prob


# -----------------------------
# Ensemble Soft Voting
# -----------------------------
prob_maps = []
for (mtype, enc) in compare_models:
    print(f"[INFO] Predicting with {mtype}|{enc}")
    prob_maps.append(get_prob_map(mtype, enc))

ensemble_prob = np.mean(prob_maps, axis=0)
ensemble_mask = (ensemble_prob > 0.5).astype(np.uint8)


# -----------------------------
# Visualization
# -----------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(orig_img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(gt_msk, cmap="gray")
plt.title("Ground Truth")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(ensemble_mask, cmap="gray")
plt.title("Ensemble Output")
plt.axis("off")

plt.tight_layout()
plt.show()