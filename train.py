'''
 # Copyright
 # 2023/4/28
 # Team: Text Analytics and Mining Lab (TAM) of South China Normal University
 # Author: Charles Yang
'''
import torch
import torch.nn as nn
import numpy as np
from load_datasets import get_dataloader
from tqdm import tqdm

def train_fn(epoch, args, data, model,optimizer,scaler, mode):
    model.train()
    train_dataloader = get_dataloader(args, data, mode=mode)
    loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), ncols=120,colour='green')

    LOSS = []
    LR = []
    for idx,batch in loop:
        # start training
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            loss = model(batch)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # record
        loss_np = loss.detach().cpu().numpy()
        LOSS.append(loss_np)
        LR.append(optimizer.param_groups[0]["lr"])

        # update the loop message
        loop.set_description(f'Epoch [{epoch + 1}/{args.epochs}] Training [{idx + 1}/{len(loop)}] Loss: {round(np.mean(LOSS),6)}')

    status = {
        'loss': np.mean(LOSS),
        'lr': LR[-1]
    }

    return status
