import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

parent_dir = r'D:\Downloads\COVID-19 CT scans'
ct_path = os.path.join(parent_dir, "ct_scans/coronacases_org_001.nii")
lung_msk_path = os.path.join(parent_dir,"lung_mask/coronacases_001.nii")
infec_msk_path = os.path.join(parent_dir,"infection_mask/coronacases_001.nii")

ct_img = nib.load(ct_path).get_fdata()
lung_mask = nib.load(lung_msk_path).get_fdata()
infec_mask = nib.load(infec_msk_path).get_fdata()

mid_slice = ct_img.shape[2] // 2
ct_slice = ct_img[:, :, mid_slice]
lung_slice = lung_mask[:, :, mid_slice]
infec_slice = infec_mask[:, :, mid_slice]

ct_slice_norm = cv2.normalize(ct_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

lung_overlay = cv2.applyColorMap((lung_slice>0).astype(np.uint8)*255, cv2.COLORMAP_BONE)
infec_overlay = cv2.applyColorMap((infec_slice>0).astype(np.uint8)*255, cv2.COLORMAP_JET)

combined_overlay = cv2.addWeighted(cv2.cvtColor(ct_slice_norm, cv2.COLOR_GRAY2BGR), 0.7, lung_overlay, 0.3, 0)
combined_overlay = cv2.addWeighted(combined_overlay, 0.7, infec_overlay, 0.3, 0)

plt.figure(figsize=(16,4))
plt.subplot(1,4,1); plt.imshow(ct_slice_norm, cmap='gray'); plt.title('CT Slice')
plt.subplot(1,4,2); plt.imshow(lung_slice, cmap='gray'); plt.title('Lung Mask')
plt.subplot(1,4,3); plt.imshow(infec_slice, cmap='gray'); plt.title('Infection Mask')
plt.subplot(1,4,4); plt.imshow(combined_overlay); plt.title('Overlay')
plt.show()
