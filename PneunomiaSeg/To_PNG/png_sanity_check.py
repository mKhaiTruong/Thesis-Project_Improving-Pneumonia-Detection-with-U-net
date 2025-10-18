import cv2
import matplotlib.pyplot as plt
import os
import random

parent_dir = r"D:\Downloads\COVID-19 CT scans"
img_dir = os.path.join(parent_dir, "ct_scans_png")
mask_dir = os.path.join(parent_dir, "infected_masks_png")

# parent_dir = r"D:\Downloads\COVID-19_lesion_seg"
# img_dir = os.path.join(parent_dir, "frames")
# mask_dir = os.path.join(parent_dir, "masks")

# === Lấy danh sách file thật có ===
img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])

# === Chọn ngẫu nhiên 10 file ===
num_samples = min(10, len(img_files))
sample_indices = random.sample(range(len(img_files)), num_samples)

for idx in sample_indices:
    img_path = os.path.join(img_dir, img_files[idx])
    mask_path = os.path.join(mask_dir, mask_files[idx])

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # === Tạo overlay ===
    overlay = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 0.7, overlay, 0.3, 0)

    # === Hiển thị ===
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(img, cmap='gray'); plt.title('CT Scan')
    plt.subplot(1, 3, 2); plt.imshow(mask, cmap='gray'); plt.title('Mask')
    plt.subplot(1, 3, 3); plt.imshow(overlay); plt.title('Overlay')
    plt.suptitle(f"Sample: {img_files[idx]}")
    plt.show()
