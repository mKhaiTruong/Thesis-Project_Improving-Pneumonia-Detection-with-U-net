from utils.metrics import compute_iou
from tqdm import tqdm
import torch

def cus_valid(model, loader, criterion, epoch, args):
    model.eval()
    val_loss, ious = 0.0, []
    
    with torch.no_grad():
        progBar = tqdm(loader, desc=f'Epoch {epoch} [VALID]', leave=False)
        for input in progBar:
            images, masks = input['image'], input['mask']
            images, masks = images.to(args.device), masks.to(args.device)
                
            outputs = model(images)
            masks   = masks.unsqueeze(1).float()
            loss = criterion(outputs, masks)
            iou = compute_iou(outputs, masks)
            val_loss += loss.item()
                
            ious.append(iou)
                
        avg_val_loss = val_loss / len(loader)
        mean_iou = sum(ious) / len(ious)
    
    return avg_val_loss, mean_iou