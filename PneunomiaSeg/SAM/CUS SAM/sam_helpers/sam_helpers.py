import torch
import torch.nn.functional as F

def prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter=False):
    
    if batched_input['point_coords'] is not None:
        points = (batched_input['point_coords'], batched_input['point_labels'])
    else:
        points = None
    
    if decoder_iter:
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points = points,
                boxes  = batched_input.get("boxes", None),
                masks  = batched_input.get("mask_inputs", None),
            )
    else:
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points = points,
            boxes  = batched_input.get("boxes", None),
            masks  = batched_input.get("mask_inputs", None),
        )
    
    low_res_masks, iou_predictions = model.mask_decoder(
        image_embeddings = image_embeddings,
        image_pe = model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings = sparse_embeddings,
        dense_prompt_embeddings  = dense_embeddings,
        multimask_output = args.multimask,
    )
    
    if args.multimask:
        max_values, max_indices = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        
        low_res = []
        for i, idx in enumerate(max_indices):
            low_res.append(low_res_masks[i:i+1, idx])
        low_res_masks = torch.stack(low_res, 0)
    
    # RESIZE OUTPUT
    masks = F.interpolate(
        low_res_masks, 
        (args.image_size, args.image_size), 
        mode="bilinear", 
        align_corners=False,
    )
    return masks, low_res_masks, iou_predictions


def to_device(batch_input, device):
    device_input = {}
    
    for key, value in batch_input.items():
        if value is not None:
            if key == 'image' or key == 'label':
                device_input[key] = value.float().to(device)
            elif type(value) is list or type(value) is torch.Size:
                device_input[key] = value
            else:
                device_input[key] = value.to(device)
        else:
            device_input[key] = value
    return device_input


import numpy as np 
from skimage.measure import label, regionprops
import torch

def setting_prompt_none(batched_input):
    batched_input['point_coords'] = None
    batched_input['point_labels'] = None
    batched_input['boxes'] = None
    return batched_input

def get_boxes_from_mask(mask, box_num=1, std=0.1, max_pixel=5):
    '''
    Args:
        mask: can be either a torch.Tensor or a numpy array or a binary mask
    Return:
        noise_boxes: noisy perturbed bounding boxes as a torch.Tensor
    '''
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
    
    label_img = label(mask)
    regions   = regionprops(label_img)
    
    # Iterate through all regions and get the bounding boxes coords
    boxes = [tuple(region.bbox) for region in regions]
    if len(boxes) == 0:
        h, w  = mask.shape
        boxes = [(0, 0, h, w)]
        
    # if n_of generated bboxes > number of categories
    # sort them by region area and select the top n regions
    elif len(boxes) >= box_num:
        sorted_regions = sorted(regions, key=lambda x: x.area, reverse=True)[:box_num]
        boxes = [tuple(region.bbox) for region in sorted_regions]
    
    # if n_of generated bboxes < number of categories
    # duplicate the existing bboxes
    elif len(boxes) < box_num:
        num_duplicates = box_num - len(boxes)
        boxes += [boxes[i % len(boxes)] for i in range(num_duplicates)]
        
    # Perturb each bounding box with noise
    noise_boxes = []
    for box in boxes:
        y0, x0, y1, x1 = box
        w, h = abs(x1 - x0), abs(y1 - y0)
        
        noise_std = min(w, h) * std
        noise_max = min(max_pixel, int(noise_std * 5))
        
        try:    noise_x = np.random.randint(-noise_max, noise_max + 1) 
        except: noise_x = 0
        try:    noise_y = np.random.randint(-noise_max, noise_max + 1) 
        except: noise_y = 0
        
        x0, y0 = x0 + noise_x, y0 + noise_y
        x1, y1 = x1 + noise_x, y1 + noise_y
        noise_boxes.append((x0, y0, x1, y1))
    
    return torch.as_tensor(noise_boxes, dtype=torch.float)

    
def init_point_sampling(mask, get_point=1):
    '''
    Initialize sample points from the mask as prompts for SAM and assign labels
    Args:
        mask (torch.Tensor): Input mask tensor
        numpoints (int): Number of points to sample (default 1)
    Returns:
        coords (torch.tensor): Tensor contains the sampling points coords (x, y)
        labels (torch.tensor): Tensor contains the corresponding labels.
        (0: background; 1: foreground) 
    '''
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
    
    fg_coords = np.argwhere(mask == 1)[:, ::-1]
    bg_coords = np.argwhere(mask == 0)[:, ::-1]
    
    fg_size, bg_size = len(fg_coords), len(bg_coords)
    
    if get_point == 1:
        if fg_size > 0:
            index = np.random.randint(fg_size)
            fg_coords = fg_coords[index]
            label = 1
        else:
            index = np.random.randint(bg_size)
            fg_coords = bg_coords[index]
            label = 0
        
        return torch.as_tensor([fg_coords.tolist()], dtype=torch.float), \
                torch.as_tensor([label], dtype=torch.int)
                
    else:
        num_fg = get_point // 2
        num_bg = get_point - num_fg
        
        fg_indices = np.random.choice(fg_size, size=num_fg, replace=True)
        bg_indices = np.random.choice(bg_size, size=num_bg, replace=True)
        fg_coords  = fg_coords[fg_indices]
        bg_coords  = bg_coords[bg_indices]
        
        coords = np.concatenate([fg_coords, bg_coords], axis=0)
        labels = np.concatenate([np.ones(num_fg), np.zeros(num_bg)]).astype(int)
        
        indices = np.random.permutation(get_point)
        
        coords = torch.as_tensor(coords[indices], dtype=torch.float)
        labels = torch.as_tensor(labels[indices], dtype=torch.int)
        
        return coords, labels
    
    
def generate_point(masks, labels, low_res_masks, batched_input, point_num):
    masks_clone   = masks.clone()
    masks_sigmoid = torch.sigmoid(masks_clone)
    masks_binary  = (masks_sigmoid > 0.5).float()
    
    low_res_masks_clone  = low_res_masks.clone()
    low_res_masks_probs  = torch.sigmoid(low_res_masks_clone)
    
    # --- create PROMPT POINTS in the INCORRECT area between GT and PRED ---
    points, point_labels =  select_random_points(masks_binary, labels, point_num)
    batched_input['mask_inputs']  = low_res_masks_probs
    batched_input['point_coords'] = torch.as_tensor(points)
    batched_input['point_labels'] = torch.as_tensor(point_labels)
    batched_input['boxes'] = None
    return batched_input
    
    
def select_random_points(pr, gt, point_num=9):
    '''
    Returns:
        batch points (np.array): Array of selected points coordinates(x, y) for each batch
        batch labels (np.array): Array of corresponding labels for each batch
            (0 - background; 1 - foreground)
    '''
    pred, gt = pr.data.cpu().numpy(), gt.data.cpu().numpy()
    error = np.zeros_like(pred)
    error[pred != gt] = 1
    
    batch_points = []
    batch_labels = []
    for j in range(error.shape[0]):
        one_pred = pred[j].squeeze(0)
        one_gt   = gt[j].squeeze(0)
        one_err  = error[j].squeeze(0)
        indices  = np.argwhere(one_err == 1)
        
        if indices.shape[0] > 0:
            selected_indices = indices[
                np.random.choice(indices.shape[0], point_num, replace=True)
            ]
        else:
            indices = np.random.randint(0, 256, size=(point_num, 2))
            selected_indices = indices[
                np.random.choice(indices.shape[0], point_num, replace=True)
            ]
            
        selected_indices = selected_indices.reshape(-1, 2)
        
        points, labels = [], []
        for i in selected_indices:
            x, y = i[0], i[1]
            
            if one_pred[x, y] == 0 and one_gt[x, y] == 1:
                label = 1
            elif one_pred[x, y] == 1 and one_gt[x, y] == 0:
                label = 0
            else: 
                label = -1

            points.append((y, x))           # Negate the coordinates
            labels.append(label)
        
        batch_points.append(points)
        batch_labels.append(labels)
    
    return np.array(batch_points), np.array(batch_labels)

def stack_dict_batch(batched_input):
    out_dict = {}
    
    for k, v in batched_input.items():
        if isinstance(v, list):
            out_dict[k] = v
        else:
            out_dict[k] = v.reshape(-1, *v.shape[2:])
    return out_dict

def postprocess_masks(low_res_masks, image_size, original_size):
    h, w = original_size
    
    masks = F.interpolate(
        low_res_masks,
        (h, w),
        mode = 'bilinear', align_corners=False,
    )
    
    if h < image_size and w < image_size:
        top  = torch.div((image_size - h), 2, rounding_mode='trunc')  
        left = torch.div((image_size - w), 2, rounding_mode='trunc') 
        masks = masks[..., top : h + top, left : w + left]
        pad  = (top, left)
    else:
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        pad = None 
        
    return masks, pad