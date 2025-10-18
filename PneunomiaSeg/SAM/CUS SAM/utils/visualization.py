import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from PIL import Image, ImageOps
import glob, os, cv2, torch
import numpy as np
import seaborn as sns
from pathlib import Path

parentDir = r"D:\LandCover"
imgDir    = os.path.join(parentDir, 'images')  
maskDir   = os.path.join(parentDir, 'masks')  
outDir    = os.path.join(parentDir, 'output')

labels_cmap = matplotlib.colors.ListedColormap(
    ["#000000", "#A9A9A9",
    "#8B8680", "#D3D3D3", "#FFFFFF"]
)

# ----- GENERAL DATASET VISUALIZATION ----
def visualize_dataset(  num_samples = 8, seed = 42,
                        w = 5, h = 6, nrows = 4, ncols = 4, 
                        save_title = None, pad = 0.8, indices = None
                    ):
    
    dataList = list(glob.glob(os.path.join(outDir, "*.jpg")))
    
    if indices == None:
        np.random.seed(seed)
        indices = np.random.randint(
            low = 0, high = len(dataList),
            size = num_samples
        )
    
    sns.set_style("white")
    
    fig, axis = plt.subplots(figsize=(h, w), nrows = num_samples//2, ncols = 4)
    for i, idx in enumerate(indices):
        r,rem = divmod(i,2)
        
        img  = cv2.imread(dataList[idx]) / 255
        
        mask = cv2.imread(
            dataList[indices[i]].split(".jpg")[0] + "_mask.png"
        )[:,:,1]
        
        axis[r,2*rem].imshow(img)
        axis[r,2*rem].set_title("Sample"+str(i+1))
        axis[r,2*rem+1].imshow(mask, cmap = labels_cmap, interpolation = None,
                            vmin = -0.5, vmax = 4.5)
        axis[r,2*rem+1].set_title("Mask" + str(i+1))
        
    plt.suptitle("Samples of 512 x 512 images", fontsize = 20)
    plt.tight_layout(pad = 0.8)
    if save_title is not None:
        plt.savefig(save_title + ".png")
    plt.show()


# ---- MEDICAL DATASET VISUALIZATION ----
'''
- Some Medical Datasets contain .tif files
'''
def visualize_tif(tif_name = None, save_title = None,
                  h = 12, w = 12, index = None):

    IMG_PATHS  = glob.glob(os.path.join(imgDir, '*.tif'))
    MASK_PATHS = glob.glob(os.path.join(maskDir, '*.tif'))
    
    if index is not None:
        img  = cv2.imread(IMG_PATHS[index]) / 255
        mask = cv2.imread(MASK_PATHS[index])
    elif tif_name is not None:
        img  = cv2.imread(os.path.join(imgDir, tif_name)) / 255
        mask = cv2.imread(os.path.join(maskDir, tif_name))
        
    labels = mask[:,:,0]
    del mask
    
    sns.set_style("white")
    fig, axis = plt.subplots(figsize = (w,h), nrows = 1, ncols =2)
    axis[0].imshow(img)
    axis[0].axis("off")
    axis[0].set_title("RGB Image")

    axis[1].imshow(
        labels, cmap=labels_cmap, interpolation = None, vmin = -0.5, vmax = 4.5
    )
    axis[1].axis("off")
    axis[1].set_title("Mask")
    
    plt.tight_layout(pad = 0.8)
    if save_title is not None:
        plt.savefig(save_title + ".png")
    plt.show()


# ---- AERIAL_UAV DATASET VISUALIZATION ----
'''
- UAV JPG images usually have EXIF orientation tag, for example: Rotate 180° 
or Rotate 90°
'''

def visualize_jpg_exif(images, masks):
    i = 0
    for img_path, mask_path in zip(images, masks):
        
        img_id  = Path(img_path).stem
        mask_id = Path(mask_path).stem
        print(f'image id = {img_id} and mask id = {mask_id}')
        
        visualize_image_mask(img_path, mask_path)
        i+=1
        if i == 10:
            break
    
def visualize_image_mask(img_path, mask_path):
    img = Image.open(img_path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    msk = Image.open(mask_path)
    
    img = img.resize(msk.size, resample=Image.BILINEAR)
    img = np.array(img)
    msk = np.array(msk)
    
    print("IMG:", img.shape, img.dtype)
    print("MASK unique values:", np.unique(msk))
    
    fig, axes = plt.subplots(1, 2, figsize=(12,6))
    axes[0].imshow(img)
    axes[0].set_title("Image")
    axes[0].axis("off")
    
    axes[1].imshow(msk, cmap='nipy_spectral', interpolation='none')
    axes[1].set_title("Mask")
    axes[1].axis("off")
    
    plt.show(block=True)

# ----- VISUALIZE IMAGE WITH BBOXES -----
def visualize_imgWithBoxes_mask(img, class_mask, labels):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(img)
    axes[0].set_title(f"Image with boxes ({labels})")
    axes[0].axis("off")
        
    axes[1].imshow(class_mask, cmap="gray")
    axes[1].set_title(f"Mask for class {labels}")
    axes[1].axis("off")
        
    plt.show()
    
    
# ---- VISUALIZE FROM JSON FILE ----
def visualize_coco_annotation(coco, img_paths, mask_paths, all_labels, num_samples=3):

    id2file = {img["id"]: img["file_name"] for img in coco["images"]}
   
    ann_dict = {}
    for ann in coco["annotations"]:
        ann_dict.setdefault(ann["image_id"], []).append(ann)

    for img_id in list(id2file.keys())[:num_samples]:
        
        file_name = id2file[img_id]
        
        img_matches = [p for p in img_paths if os.path.basename(p) == file_name]
        if not img_matches:
            print(f"[WARN] Can't find the image with {file_name}")
            continue
        img_path = img_matches[0]
        
        msk_matches = [m for m in mask_paths if os.path.basename(m).startswith(file_name.split('.')[0])]
        if not msk_matches:
            print(f"[WARN] Can't find the mask with {file_name}")
            continue
        msk_path = msk_matches[0]

        img  = cv2.imread(img_path)
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(msk_path, cv2.IMREAD_UNCHANGED)

        anns = ann_dict.get(img_id, [])
        present_classes = sorted(set([anno["category_id"] for anno in anns]))
        for cate_id in present_classes:
            img_orig = img.copy()
            filtered_annos = [anno for anno in anns if anno["category_id"] == cate_id]
            
            label_ids = list({anno['label_id'] for anno in filtered_annos})
            # print(label_ids)
            selected_labels = [all_labels[i] for i in label_ids]
            
            for anno in filtered_annos:
                x, y, w, h = anno["bbox"]
                cv2.rectangle(img_orig, (x, y), (x+w, y+h), (255,0,0), 10)
            
            class_mask = np.isin(mask, label_ids).astype('uint8') * 255
            visualize_imgWithBoxes_mask(img_orig, class_mask, selected_labels)
        
        

# ----- FROM-TENSOR-BACK-TO-NORMAL VISUALIZATION ----
'''
- After fed into Dataset, images are usually normalized for model to learn. So in 
order to visualize, we need to denormalize them first.
'''
def denormalize_image(img_tensor, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    img = img_tensor.clone().cpu().numpy()
    
    if img.ndim == 4:
        img = img.squeeze(0)  # [1,C,H,W] -> [C,H,W]

    if img.ndim == 2:        # grayscale
        return (img * 255).astype(np.uint8)
    elif img.shape[0] == 3:  # [C,H,W] -> [H,W,C]
        img = np.transpose(img, (1,2,0))
    
    for i in range(3):
        img[i] = img[i] * std[i] + mean[i]
    img = img.astype(np.uint8)
    img = np.transpose(img, (1,2,0))  # C,H,W -> H,W,C
    return img
def visualize_sample(img_tensor, mask_tensor, name=None, class_colors=None):
    img = denormalize_image(img_tensor)
    mask = mask_tensor.squeeze().numpy()     # 1,H,W → H,W
    
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(img)
    if name:
        plt.title(f"Image {name}") 
    else: plt.title("Image")
    plt.axis("off")
    
    plt.subplot(1,2,2)
    if class_colors:
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for i, color in enumerate(class_colors):
            colored_mask[mask == i] = color
        plt.imshow(colored_mask)
    else:
        plt.imshow(mask, cmap='nipy_spectral', interpolation='none')
    plt.title("Mask")
    plt.axis("off")
    plt.show()
def visualize_sample_detection(img_tensor, target, class_names=None):
    # de-normalize
    img = denormalize_image(img_tensor)   # (H,W,3)
    
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    
    boxes  = target["boxes"].cpu().numpy()
    labels = target["labels"].cpu().numpy()
    
    for box, label in zip(boxes, labels):
        x1,y1,x2,y2 = box
        rect = patches.Rectangle((x1,y1), x2-x1, y2-y1, linewidth=2,
                                 edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
        if class_names:
            plt.text(x1, y1-5, class_names[label-1], color="yellow", fontsize=12)
    plt.axis("off")
    plt.show()


# ---- VISUALIZE ORIGINAL IMAGE AND PREDICTED OUTPUTS ----
'''
- original image is just plain image from your computer
- predicted outputs: can be masks, can be bboxes.
'''
def visualize_bboxes(orig_img, boxes, labels=None, scores=None, score_thresh=0.5):
    """
    Visualize bounding boxes on the original image.
    
    Args:
        orig_img: np.array, original image (H x W x 3)
        boxes: np.array of shape (N, 4), predicted boxes (x1, y1, x2, y2)
        labels: list or array of labels, optional
        scores: list or array of confidence scores, optional
        score_thresh: float, only visualize boxes with score >= threshold
    """
    H, W = orig_img.shape[:2]
    boxes[:, [0, 2]] *= (W / 512)
    boxes[:, [1, 3]] *= (H / 512)
    
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(orig_img)
    
    for i, box in enumerate(boxes):
        if scores is not None and scores[i] < score_thresh:
            continue
        
        x1, y1, x2, y2 = box
        # Ensure coordinates are integers and within image bounds
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, W-1), min(y2, H-1)
        
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                         fill=False, color='red', linewidth=2)
        ax.add_patch(rect)
        
        if labels is not None and scores is not None:
            ax.text(x1, y1-5, f"{labels[i]}:{scores[i]:.2f}",
                    color='yellow', fontsize=8, backgroundcolor="black")
    
    plt.axis('off')
    plt.show()


def visualize_masks(orig_img, masks, colors=None, alpha=0.4):
    """
    Visualize segmentation masks on the original image.
    
    Args:
        orig_img: np.array, original image (H x W x 3)
        masks: list or array of masks, each mask same size as image (H x W)
        colors: list of colors for each mask, optional
        alpha: float, transparency of masks
    """
    H, W = orig_img.shape[:2]
    
    overlay = orig_img.copy()
    
    for i, mask in enumerate(masks):
        # Ensure mask same size as image
        if mask.shape != (H, W):
            mask = np.array(Image.fromarray(mask).resize((W, H), resample=Image.NEAREST))
        
        color = colors[i] if colors is not None else np.random.randint(0, 255, size=3)
        color_mask = np.zeros_like(orig_img, dtype=np.uint8)
        color_mask[mask > 0] = color
        overlay = (overlay * (1-alpha) + color_mask * alpha).astype(np.uint8)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(overlay)
    plt.axis('off')
    plt.show()

# ---- VISUALIZATION WITH PREDICTED MASKS
def denormalize_image(img_tensor, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
    if img_tensor.ndim == 4:
        img_tensor = img_tensor.squeeze(0)
    
    img = img_tensor.detach().cpu().numpy()  # [C,H,W] hoặc [H,W]

    # Nếu grayscale
    if img.ndim == 2:
        img = (img * 255).astype(np.uint8)
        return img

    # Nếu color
    if img.shape[0] == 3:  # [C,H,W] -> [H,W,C]
        img = np.transpose(img, (1,2,0))

    # Denormalize
    img = img * np.array(std) + np.array(mean)
    img = np.clip(img*255.0, 0, 255).astype(np.uint8)
    return img

def visualize_msk_prediction(orig_img, mask_pred, alpha=0.5):
    """
    Visualize original image, predicted mask, and overlay.
    
    Args:
        orig_img (np.array): Original image (H,W,3), uint8.
        mask_pred (torch.Tensor or np.array): Model output mask (1,H,W) or (H,W).
        alpha (float): Transparency for overlay.
    """
    if isinstance(orig_img, torch.Tensor):
        if orig_img.dim() == 4 and orig_img.size(0) == 1:
            orig_img = orig_img.squeeze(0)   # (1,3,H,W) → (3,H,W)
        orig_img = denormalize_image(orig_img)
    
    mask_bin = safe_mask_to_2d(mask_pred)

    # make overlay
    overlay = orig_img.copy()
    color = np.array([0, 255, 0], dtype=np.uint8)
    overlay[mask_bin == 1] = (
        overlay[mask_bin == 1] * (1 - alpha) + color * alpha
    ).astype(np.uint8)

    # plot
    plt.figure(figsize=(15,10))
    plt.subplot(1,3,1)
    plt.imshow(orig_img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(mask_bin, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis("off")

    plt.show()
def safe_mask_to_2d(mask_pred):
    """ Convert mask tensor/array to 2D (H,W) safely """
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    mask = np.array(mask)
    if mask.ndim == 3:
        if mask.shape[0] == 1:       # (1,H,W)
            mask = mask[0]
        elif mask.shape[2] == 1:     # (H,W,1)
            mask = mask[:,:,0]

    return mask.astype(np.uint8)

if __name__ == '__main__':
    visualize_dataset()