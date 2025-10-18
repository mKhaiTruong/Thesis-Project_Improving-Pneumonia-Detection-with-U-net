# -------------------------------------------------------------------------------------
# FROM .NII FILES TO .PNG FILES
# -------------------------------------------------------------------------------------
import nibabel as nib
import numpy as np
import cv2
import os
from tqdm import tqdm
import random

# === PATH CONFIG ===
parent_dir = r"D:\Downloads\COVID-19 CT scans"
ct_dir = os.path.join(parent_dir, "ct_scans")
lung_mask_dir = os.path.join(parent_dir, "lung_mask")
infec_mask_dir = os.path.join(parent_dir, "infection_mask")

out_img_dir = os.path.join(parent_dir, "ct_scans_png")
out_mask_dir = os.path.join(parent_dir, "infected_masks_png")

os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_mask_dir, exist_ok=True)

ct_files = sorted([f for f in os.listdir(ct_dir) if f.endswith(".nii")])
lung_files = sorted([f for f in os.listdir(lung_mask_dir) if f.endswith(".nii")])
infec_files = sorted([f for f in os.listdir(infec_mask_dir) if f.endswith(".nii")])

img_counter = 1

for ct_file, lung_file, infec_file in tqdm(zip(ct_files, lung_files, infec_files), total=len(ct_files), desc="Processing NIfTI → PNG"):
    ct_path = os.path.join(ct_dir, ct_file)
    lung_path = os.path.join(lung_mask_dir, lung_file)
    infec_path = os.path.join(infec_mask_dir, infec_file)

    ct_img = nib.load(ct_path).get_fdata()
    lung_mask = nib.load(lung_path).get_fdata()
    infec_mask = nib.load(infec_path).get_fdata()
    assert ct_img.shape == lung_mask.shape == infec_mask.shape

    for i in range(ct_img.shape[2]):
        if i % 4 != 0:
            continue

        ct_slice = ct_img[:, :, i]
        lung_slice = lung_mask[:, :, i]
        infec_slice = infec_mask[:, :, i]

        if not np.any(infec_slice > 0):
            if random.random() > 0.2:
                continue

        mask_bin = (infec_slice > 0).astype(np.uint8) * 255

        if np.any(lung_slice > 0):
            ys, xs = np.where(lung_slice > 0)
            y1, y2 = ys.min(), ys.max() + 1
            x1, x2 = xs.min(), xs.max() + 1

            # Diện tích phổi và padding scale
            lung_area = (y2 - y1) * (x2 - x1)
            img_area = lung_slice.shape[0] * lung_slice.shape[1]
            
            # dynamic padding nhưng giới hạn max crop size
            pad_scale = min(max(lung_area / img_area * 0.3, 0.05), 0.35)  # max 0.35 để tránh crop quá lớn

            h_pad = int((y2 - y1) * pad_scale)
            w_pad = int((x2 - x1) * pad_scale)

            # tính crop với giới hạn max/min để không mất phổi
            y1_new = max(y1 - h_pad, 0)
            y2_new = min(y2 + h_pad, ct_slice.shape[0])
            x1_new = max(x1 - w_pad, 0)
            x2_new = min(x2 + w_pad, ct_slice.shape[1])

            # nếu crop quá nhỏ (ví dụ phổi quá bé), mở rộng tối thiểu 50% khung gốc
            min_crop_h = int(0.5 * ct_slice.shape[0])
            min_crop_w = int(0.5 * ct_slice.shape[1])
            
            if (y2_new - y1_new) < min_crop_h:
                center_y = (y1 + y2) // 2
                y1_new = max(center_y - min_crop_h // 2, 0)
                y2_new = min(y1_new + min_crop_h, ct_slice.shape[0])
            
            if (x2_new - x1_new) < min_crop_w:
                center_x = (x1 + x2) // 2
                x1_new = max(center_x - min_crop_w // 2, 0)
                x2_new = min(x1_new + min_crop_w, ct_slice.shape[1])

            ct_slice = ct_slice[y1_new:y2_new, x1_new:x2_new]
            mask_bin = mask_bin[y1_new:y2_new, x1_new:x2_new]
        else:
            ct_slice = ct_slice
            mask_bin = mask_bin

        lung_pixels = ct_slice[mask_bin > 0]
        if lung_pixels.size > 0:
            min_val = lung_pixels.min()
            max_val = lung_pixels.max()
        else:
            min_val, max_val = ct_slice.min(), ct_slice.max()
        
        ct_norm = np.clip(ct_slice, min_val, max_val)
        ct_norm = ((ct_norm - min_val) / (max_val - min_val + 1e-5) * 255).astype(np.uint8)
        ct_norm = cv2.resize(ct_norm, (512, 512), interpolation=cv2.INTER_NEAREST)
        ct_display = cv2.applyColorMap(ct_norm, cv2.COLORMAP_BONE)
        
        mask_norm = cv2.resize(mask_bin, (512, 512), interpolation=cv2.INTER_NEAREST)
        mask_norm = mask_norm.astype(np.uint8)

        cv2.imwrite(os.path.join(out_img_dir, f"scan_{img_counter:05d}.png"), ct_norm)
        cv2.imwrite(os.path.join(out_mask_dir, f"mask_{img_counter:05d}.png"), mask_norm)
        img_counter += 1

print(f"\n✅ Done! Saved {img_counter - 1} PNG pairs.")

# -------------------------------------------------------------------------------------
# CREATE ADDITIONAL DATASETS (COVID-19 CT scans 2 & 3)
# -------------------------------------------------------------------------------------
import shutil, random

# === Function to clone subset sequentially ===
def create_additional_dataset(src_parent, dst_parent, portion):
    src_img_dir = os.path.join(src_parent, "ct_scans_png")
    src_mask_dir = os.path.join(src_parent, "infected_masks_png")

    dst_img_dir = os.path.join(dst_parent, "ct_scans_png")
    dst_mask_dir = os.path.join(dst_parent, "infected_masks_png")

    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_mask_dir, exist_ok=True)

    images = sorted([f for f in os.listdir(src_img_dir) if f.endswith(".png")])
    masks = sorted([f for f in os.listdir(src_mask_dir) if f.endswith(".png")])
    assert len(images) == len(masks), "❌ Image-mask count mismatch!"

    total = len(images)
    subset_size = int(total * portion)
    start_idx = random.randint(0, total - subset_size - 1)
    selected = range(start_idx, start_idx + subset_size)

    print(f"\n📁 Creating {dst_parent}: taking {subset_size}/{total} pairs (from index {start_idx})")

    for i in tqdm(selected, desc=f"Copying to {os.path.basename(dst_parent)}"):
        shutil.copy2(os.path.join(src_img_dir, images[i]), os.path.join(dst_img_dir, images[i]))
        shutil.copy2(os.path.join(src_mask_dir, masks[i]), os.path.join(dst_mask_dir, masks[i]))

    print(f"✅ Done creating {dst_parent}!\n")


# === Main process ===
src_dataset = r"D:\Downloads\COVID-19 CT scans"
datasets_to_create = [
    (r"D:\Downloads\COVID-19 CT scans 2", 0.5),
    (r"D:\Downloads\COVID-19 CT scans 3", 0.5),
]

for dst, portion in datasets_to_create:
    create_additional_dataset(src_dataset, dst, portion)