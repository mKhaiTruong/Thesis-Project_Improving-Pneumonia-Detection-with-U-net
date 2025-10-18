import numpy as np
import cv2, os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def load_mask(path, size=(256, 256)):
    mask = cv2.imread(path, 0)  # grayscale
    mask = (mask > 127).astype(np.uint8)  # threshold
    if size is not None:
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_LINEAR)
    return mask

def iou_score(gt, pred):
    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    if union == 0: return 1.0  # empty mask
    return intersection / union

def dice_score(gt, pred):
    intersection = np.logical_and(gt, pred).sum()
    total = gt.sum() + pred.sum()
    if total == 0: return 1.0
    return 2 * intersection / total

img_dir = r"D:\Downloads\COVID-19 CT scans 3\ct_scans_png"
gt_dir = r"D:\Downloads\COVID-19 CT scans 3\infected_masks_png"
unet_dir = r"D:\Downloads\COVID-19 CT scans 3\unet_unetpp_b3_22epochs_masks"
sam_dir = r"D:\Downloads\COVID-19 CT scans 3\sam_adapter_masks"

iou_unet_list = []
iou_sam_list = []
dice_unet_list = []
dice_sam_list = []

for fname in os.listdir(gt_dir):
    file_id = os.path.splitext(fname)[0].split('_')[-1]  # "00512"
    
    gt_mask = load_mask(os.path.join(gt_dir, fname))
    
    unet_mask = load_mask(os.path.join(unet_dir, f"scan_{file_id}.png"),
                          size=(gt_mask.shape[1], gt_mask.shape[0]))
    sam_mask = load_mask(os.path.join(sam_dir, f"scan_{file_id}.png"),
                         size=(gt_mask.shape[1], gt_mask.shape[0]))

    iou_unet_list.append(iou_score(gt_mask, unet_mask))
    iou_sam_list.append(iou_score(gt_mask, sam_mask))
    dice_unet_list.append(dice_score(gt_mask, unet_mask))
    dice_sam_list.append(dice_score(gt_mask, sam_mask))

# Tạo dataframe
df_metrics = pd.DataFrame({
    "IoU": iou_sam_list + iou_unet_list,
    "Dice": dice_sam_list + dice_unet_list,
    "Model": ["U-net + SAM-Adapter"]*len(iou_sam_list) + ["Unet"]*len(iou_unet_list)
})

sns.set_theme(style="whitegrid", font_scale=1.2)
palette = {"Unet": "#FF3F7F", "U-net + SAM-Adapter": "#8C00FF"}

def plot_metric(metric_name):
    plt.figure(figsize=(8, 3.5))
    
    ax = sns.violinplot(
        y="Model", x=metric_name, data=df_metrics,
        palette=palette, inner=None, linewidth=1.2, cut=0, bw=0.2
    )
    sns.swarmplot(
        y="Model", x=metric_name, data=df_metrics,
        color="k", alpha=0.5, size=1, ax=ax
    )

    # Mean line
    means = df_metrics.groupby("Model")[metric_name].mean()
    print(means)
    for i, (_, mean_val) in enumerate(means.items()):
        plt.scatter(mean_val, i, color="gold", s=100, edgecolor="black", zorder=10)
        plt.text(mean_val + 0.002, i + 0.05, f"{mean_val:.3f}", fontsize=10, color="black")
    
    plt.title(f"{metric_name} Comparison", weight="bold")
    plt.xlabel(f"{metric_name} Score")
    plt.ylabel("")
    plt.xlim(0.2, 1.0)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

plot_metric("IoU")
plot_metric("Dice")

def overlay_mask(img, mask, color=(255,0,0), alpha=0.5):
    """Overlay binary mask on image"""
    img = img.copy()
    mask_color = np.zeros_like(img)
    mask_color[..., 0] = color[0]
    mask_color[..., 1] = color[1]
    mask_color[..., 2] = color[2]
    overlay = np.where(mask[..., None], (1-alpha)*img + alpha*mask_color, img)
    return overlay.astype(np.uint8)

sample_files = list(os.listdir(gt_dir))[:50]
for fname in sample_files:
    file_id = os.path.splitext(fname)[0].split('_')[-1]
    
    img_path = os.path.join(img_dir, f"scan_{file_id}.png")
    sam_path = os.path.join(sam_dir, f"scan_{file_id}.png")
    net_path = os.path.join(unet_dir, f"scan_{file_id}.png")
    
    
    
    if not os.path.exists(img_path):
        print(f"Warning: {img_path} not found, skipping")
        continue
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gt_mask = load_mask(os.path.join(gt_dir, fname), size=(img.shape[1], img.shape[0]))
    unet_mask = load_mask(net_path, size=(img.shape[1], img.shape[0]))
    sam_mask  = load_mask(sam_path, size=(img.shape[1], img.shape[0]))

    fig, axes = plt.subplots(1, 4, figsize=(20,5))

    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(overlay_mask(img, gt_mask, color=(255,0,0), alpha=0.5))
    axes[1].set_title("Original + GT Mask")
    axes[1].axis("off")

    axes[2].imshow(overlay_mask(img, unet_mask, color=(65, 166, 126), alpha=0.5))
    axes[2].set_title("Original + Unet Mask")
    axes[2].axis("off")

    axes[3].imshow(overlay_mask(img, sam_mask, color=(255, 217, 61), alpha=0.75))
    axes[3].set_title("Original + Unet + SAM-Adapter Mask")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()