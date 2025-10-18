from tqdm import tqdm
import torch

def train_one_epoch(model, loader, criterion, optimizer, scheduler, epoch, args):
    model.train()
    total_loss = 0.0
        
    progBar = tqdm(loader, desc=f'Epoch {epoch} [TRAIN]', leave=False)
    for input in progBar:
        images, masks = input['image'], input['mask']
        images, masks = images.to(args.device), masks.to(args.device)
        optimizer.zero_grad()
            
        outputs = model(images)
        masks   = masks.unsqueeze(1).float()
        
        loss    = criterion(outputs, masks)
        if input['source'] == 'Domain Based':
            loss *= 2.0
        loss.backward()
        optimizer.step()
        scheduler.step()
            
        total_loss += loss.item()
        
    avg_train_loss = total_loss / len(loader)
    return avg_train_loss