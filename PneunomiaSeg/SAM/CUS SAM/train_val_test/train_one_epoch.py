import sys, os, random
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import torch.nn.functional as F

sys.path.append(r"..\PneunomiaSeg\SAM\CUS SAM")
from utils.metrics import SegMetrics
from sam_helpers.sam_helpers import *
    
    
def train_one_epoch(args, model, optimizer, train_loader, epoch, criterion):
    
    scaler = GradScaler(enabled=args.use_amp)
    train_losses = []
    train_iter_metrics = [0] * len(args.metrics)
    model.train()
    
    progBar = tqdm(train_loader, desc=f'Epoch {epoch} [TRAIN]', leave=False)
    for batch, batched_input in enumerate(progBar):
        batched_input = stack_dict_batch(batched_input)
        batched_input = to_device(batched_input, args.device)
        
        if random.random() > 0.5:
            batched_input['point_coords'] = None
            flag = 'boxes'
        else:
            batched_input['boxes'] = None
            flag = 'points'

        # Stage 1: only open grad_ for Adapter inside IMAGE_ENCODER
        for n, p in model.image_encoder.named_parameters():
            p.requires_grad = ('Adapter' in n)
        
        
        # ---- STAGE 1 FORWARD + BACKWARD (encode images + decoder) ----
        labels = batched_input['label']
        
        with autocast(enabled=args.use_amp):
            image_embeddings = model.image_encoder(batched_input["image"])

            B, _, _, _ = image_embeddings.shape
            image_embeddings_repeat = []
            
            for i in range(B):
                image_embed = image_embeddings[i]
                image_embed = image_embed.repeat(args.mask_num, 1, 1, 1)
                image_embeddings_repeat.append(image_embed)
            image_embeddings = torch.cat(image_embeddings_repeat, dim=0)
            
            masks, low_res_masks, iou_predictions = prompt_and_decoder(
                args, batched_input, model, image_embeddings, decoder_iter=False
            )
            loss = criterion(masks, labels, iou_predictions)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        # if int(batch + 1) % 50 == 0:
        #     print(f'Epoch {epoch} at Batch {batch+1} | '
        #           f'first {flag} prompt = {SegMetrics(masks, labels, args.metrics)}')
        
        
        # ---- INTERACTIVE LOOP "SELF-CORRECT" ----
        # Generating prompts from the wrong predictions -> retrain -> better performance
        point_num = random.choice(args.point_list)
        batched_input = generate_point(masks, labels, low_res_masks, batched_input, point_num)
        batched_input = to_device(batched_input, args.device)
        
        
        # ---- Stage 2: free image_encoder + train the rests (prompt+decoder) multiple times ----
        image_embeddings = image_embeddings.detach().clone()
        for n, p in model.named_parameters():
            p.requires_grad = ('image_encoder' not in n)
        
        init_mask_num = np.random.randint(1, max(2, args.iter_point - 1))
        for iter in range(args.iter_point):
            
            # randomly train without prompts
            if iter == init_mask_num or iter == args.iter_point - 1:
                batched_input = setting_prompt_none(batched_input)
            
            with autocast(enabled=args.use_amp):
                masks, low_res_masks, iou_predictions = prompt_and_decoder(
                    args, batched_input, model, image_embeddings, decoder_iter=True
                )
                loss = criterion(masks, labels, iou_predictions)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            
            # if not the last iteration, again, create prompts from the incorrect predictions
            if iter != args.iter_point - 1:
                point_num = random.choice(args.point_list)
                batched_input = generate_point(masks, labels, low_res_masks, batched_input, point_num)
                batched_input = to_device(batched_input, args.device)
            
            
            # if int(batch + 1) % 50 == 0:
            #     if iter == init_mask_num or iter == args.iter_point - 1:
            #         print(
            #             f'Epoch: {epoch} at batch {batch + 1} | '
            #             f'mask prompt = {SegMetrics(masks, labels, args.metrics)}'
            #         )
            #     else:
            #         print(
            #             f'Epoch: {epoch} at batch {batch + 1} | '
            #             f'point = {point_num}, prompt = {SegMetrics(masks, labels, args.metrics)}'
            #         )
        
        # if int(batch + 1) % 200 == 0:
        #     print(f'Epoch: {epoch} iteration {batch + 1} | loss = {loss.item()}')
        #     save_path = os.path.join(f"{args.work_dir}/models", args.run_name, f'epoch{epoch+1}_batch{batch+1}_sam.pth')
        #     state = {'model': model.state_dict(), 'optimizer': optimizer}
        #     torch.save(state, save_path)
        
        train_losses.append(loss.item())
        
        gpu_info = {}
        gpu_info['gpu_name'] = args.device
        # progBar.set_postfix(train_loss=loss.item(), gpu_info=gpu_info)
        
        train_batch_metrics = SegMetrics(masks, labels, args.metrics)
        train_iter_metrics  = [train_iter_metrics[i] + train_batch_metrics[i] for i in range(len(args.metrics))]
    
    return train_losses, train_iter_metrics