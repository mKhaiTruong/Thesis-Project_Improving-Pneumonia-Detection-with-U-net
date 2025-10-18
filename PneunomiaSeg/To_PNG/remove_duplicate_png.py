# -------------------------------------------------------------------------------------
# SAMPLE EVERY 4 PNG IMAGES — REDUCE DUPLICATION
# -------------------------------------------------------------------------------------
import os
import shutil
from tqdm import tqdm

# === PATH CONFIG ===
src_parent = r"D:\Downloads\COVID-19_lesion_seg"
src_img_dir = os.path.join(src_parent, "frames")
src_mask_dir = os.path.join(src_parent, "masks")

dst_parent = os.path.join(src_parent, "subset_step4_skip100")
dst_img_dir = os.path.join(dst_parent, "frames")
dst_mask_dir = os.path.join(dst_parent, "masks")

os.makedirs(dst_img_dir, exist_ok=True)
os.makedirs(dst_mask_dir, exist_ok=True)

# === FILES (sorted to keep pairing) ===
images = ([f for f in os.listdir(src_img_dir) if f.lower().endswith(".png")])
masks = ([f for f in os.listdir(src_mask_dir) if f.lower().endswith(".png")])

assert len(images) == len(masks), "❌ Số lượng ảnh và mask không khớp!"
total = len(images)
print(f"📂 Found {total} PNG pairs total.")

# === PARAMETERS ===
first_n = 100   # số ảnh đầu tiên sẽ giữ nguyên
step = 5        # các ảnh còn lại lấy mỗi step-th ảnh

# 1) Copy first_n images
for i in tqdm(range(first_n), desc="Copying first 100 images"):
    shutil.copy2(os.path.join(src_img_dir, images[i]), os.path.join(dst_img_dir, images[i]))
    shutil.copy2(os.path.join(src_mask_dir, masks[i]), os.path.join(dst_mask_dir, masks[i]))

# 2) Copy every `step`-th image starting from index first_n
for i in tqdm(range(first_n, total, step), desc="Copying every 4th image after first 100"):
    shutil.copy2(os.path.join(src_img_dir, images[i]), os.path.join(dst_img_dir, images[i]))
    shutil.copy2(os.path.join(src_mask_dir, masks[i]), os.path.join(dst_mask_dir, masks[i]))

copied_total = len(os.listdir(dst_img_dir))
print(f"\n✅ Done! Created '{dst_parent}' with {copied_total} image-mask pairs.")