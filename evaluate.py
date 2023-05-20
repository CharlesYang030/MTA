'''
 # Copyright
 # 2023/4/28
 # Team: Text Analytics and Mining Lab (TAM) of South China Normal University
 # Author: Charles Yang
'''
import torch
import torch.nn.functional as F
import numpy as np
from load_datasets import get_dataloader
from utils import cal_metrics
from tqdm import tqdm
import time

@torch.no_grad()
def eval_module(model,sentence,img_names,candidate_images,mode):
    text_encoder = model.mtext_encoder

    sentence_feats = text_encoder(model.clip_tokenize(sentence, truncate=True).to(model.device))
    sentence_feats = F.normalize(sentence_feats, dim=-1)

    this_bs = len(sentence)  # 记录当前batch的长度
    pred_imgs = []
    sort_ten = []
    for i in range(this_bs):
        # 得到每个歧义短语对应的10张候选图片的特征
        candidate_img_feats = model.vision_encoder(candidate_images[i].to(model.device))
        candidate_img_feats = F.normalize(candidate_img_feats, dim=-1)

        # 得到单个歧义短语的特征
        per_sentence_feats = sentence_feats[i].unsqueeze(0)

        # 计算logits
        sim_logits = per_sentence_feats @ candidate_img_feats.T  # 计算每个歧义短语和10张候选图片的相似度矩阵
        final_logits = sim_logits

        # 确定预测图片
        logits_numpy = final_logits.softmax(1).detach().cpu().numpy()
        max_index = np.argmax(logits_numpy)
        pred = img_names[i][max_index]
        pred_imgs.append(pred)

        # 记录结果
        _, idx_topk = torch.topk(final_logits, k=10, dim=-1)
        result = []
        for j in idx_topk[0]:
            j = int(j)
            result.append(img_names[i][j])
        sort_ten.append(result)

    return pred_imgs, sort_ten

@torch.no_grad()
def evaluation_fn(epoch, args, data, model,mode,input_pattern):
    model.eval()
    eval_dataloader = get_dataloader(args, data, mode=mode, input_pattern=input_pattern)
    colour = get_colour(input_pattern)
    loop = tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), ncols=120,colour=colour)

    count = 0
    acc_sum = 0
    GOLD = []
    best_imgs = []
    SORT10 = []

    for idx,batch in loop:
        amb,sentence,img_names,candidate_images,gold_img = batch
        GOLD.append(gold_img)

        # start evaluating
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred_imgs, sort_ten = eval_module(model,sentence,img_names,candidate_images,mode=mode)

        ### record
        best_imgs += pred_imgs
        SORT10.append(sort_ten)

        # calculate the accuracy
        for i, pred in enumerate(pred_imgs):
            if pred == gold_img[i]:
                acc_sum += 1

        count += len(sentence)
        now_acc,now_mrr = cal_metrics(SORT10, GOLD, count)

        # update the loop message
        loop.set_description(f'{mode} Epoch [{epoch + 1}/{args.epochs}] Evaluating [{idx + 1}/{len(loop)}] Acc:{now_acc:.4f} Mrr:{now_mrr:.4f}')

    test_mode = ['test_en','test_fa','test_it']
    # calculate metrics
    acc, mrr = cal_metrics(SORT10, GOLD, count)
    time.sleep(0.1)
    if mode in test_mode:
        print(f'"{mode}" Evaluating Accuracy = ', acc, ' MRR = ', mrr)
    else:
        print('Evaluating Accuracy = ', acc, ' MRR = ', mrr)

    status = {
        'acc': acc,
        'mrr': mrr
    }

    return status,best_imgs,SORT10,GOLD

def get_colour(input_pattern):
    if input_pattern == 1:
        return 'red'
    elif input_pattern == 2:
        return 'yellow'
    elif input_pattern == 3:
        return 'blue'
