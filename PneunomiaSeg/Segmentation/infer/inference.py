import sys
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(r"..\PneunomiaSeg\SAM\CUS SAM")
from helpers import *
from dataset.unet_dataset import Test_UnetDataset

# -----------------------------
# Config
# -----------------------------
model_types   = ['dpt', 'deeplabv3plus', 'unetpp', 'segformer']
encoder_types = ['efficientnet-b3', 'mit_b1']
exp_name      = 'exp1'

# list of existing trained models
compare_models = [
    ("deeplabv3plus", "efficientnet-b3"),
    ("deeplabv3plus", "mit_b1"),
    ("segformer",     "efficientnet-b3"),
    ("segformer",     "mit_b1"),
    ("unetpp",        "efficientnet-b3"),
]

img_pth = [r"D:\Downloads\COVID-19 CT scans 3\ct_scans_png\scan_00530.png"]
msk_pth = r"D:\Downloads\COVID-19 CT scans 3\infected_masks_png\mask_00530.png"
model_dir = r"D:\Deep_Learning_Object_Detection\randProjects\PneunomiaSeg\Segmentation\results\models"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

gt_msk   = np.array(Image.open(msk_pth).convert("L"))
gt_msk   = (gt_msk > 127).astype(np.uint8)

test_ds     = Test_UnetDataset(img_pth)
img_to_pred = test_ds[0]['image'].unsqueeze(0).to(device)
img_to_show = test_ds._load_image(path=img_pth[0])

# -----------------------------
# Visualization
# -----------------------------
n_models = len(compare_models)
fig, axs = plt.subplots(n_models, 3, figsize=(8, 2*n_models))

for i, (mtype, enc) in enumerate(compare_models):
    # Original
    axs[i,0].imshow(img_to_show)
    axs[i,0].set_title("Original")
    axs[i,0].axis("off")

    # GT
    axs[i,1].imshow(gt_msk, cmap="gray")
    axs[i,1].set_title("Ground Truth")
    axs[i,1].axis("off")

    # Prediction
    pred_msk = get_prediction(model_dir, mtype, enc, exp_name, img_to_pred)
    axs[i,2].imshow(pred_msk, cmap="gray")
    centers = mask_to_centers(pred_msk)
    for (cx, cy) in centers:
        axs[i,2].scatter(cx, cy, c='red', s=24, marker='x')
    axs[i,2].set_title(f"Predicted: {mtype}|{enc}")
    axs[i,2].axis("off")

plt.tight_layout()
plt.show()