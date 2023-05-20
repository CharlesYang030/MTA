import random
import os
from datetime import datetime
import numpy as np
import torch
import math

from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch)) + min_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def _convert_image_to_rgb(image):
    return image.convert("RGB")

_transform = Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

# 冻住模型指定的层数
def frozen_model(model,mode,open):
    for name, value in model.named_parameters():
        value.requires_grad = False
    if mode == 'partial':
        ### 冻结clip中的某些层
        grad_noupdate = []
        for i in range(12-open,12):
            grad_noupdate.append('mtext_encoder.transformer.resblocks.'+str(i))
        for name, value in model.named_parameters():
            title = name.split('.')[0]
            if title == 'mtext_encoder':
                flag = False
                for g in grad_noupdate:
                    if g in name:
                        flag = True
                        break
                if not (name in grad_noupdate or flag):
                    value.requires_grad = False
                else:
                    value.requires_grad = True
    elif mode == 'none':
        for name, value in model.named_parameters():
            value.requires_grad = True

# 输出模型参数情况
def display_modelpara(model):
    for name, param in model.named_parameters():
        print(name, param.requires_grad)

def calculate_modelpara(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print({'Total': total_num, 'Trainable': trainable_num})

def convert_models_to_fp32(model):
    for p in model.parameters():
        if not p.data == None:
            p.data = p.data.float()
        if not p.grad == None:
            p.grad.data = p.grad.data.float()

def cal_metrics(SORT10,GOLD,total):
    gold_imgs = []
    for golds in GOLD:
        for g in golds:
            gold_imgs.append(g)

    acc = 0
    mrr = 0
    count = 0
    for prediction in SORT10:
        batch = len(prediction)
        for i in range(batch):
            gold = gold_imgs[count + i]
            pred_best = prediction[i][0]
            # acc
            if gold == pred_best:
                acc += 1
            # mrr
            for j in range(len(prediction[i])):
                if gold == prediction[i][j]:
                    # 注意j的取值从0开始
                    mrr += 1/(j+1)
                    break
        count += batch
    acc = acc / total * 100.
    mrr = mrr / total * 100.
    return acc,mrr

def integrate_test_status(SORT10S,GOLDS):
    total_gold_imgs = []
    for GOLD in GOLDS:
        for golds in GOLD:
            for g in golds:
                total_gold_imgs.append(g)
    total_length = len(total_gold_imgs)

    total_sort10 = []
    for SORT10 in SORT10S:
        for sort in SORT10:
            for s10 in sort:
                total_sort10.append(s10)

    # calculate acc and mrr
    acc = 0
    mrr = 0
    for idx,sort10 in enumerate(total_sort10):
        pred_best = sort10[0]
        gold = total_gold_imgs[idx]
        # acc
        if gold == pred_best:
            acc += 1
        # mrr
        for j in range(len(sort10)):
            if gold == sort10[j]:
                # 注意j的取值从0开始
                mrr += 1 / (j + 1)
                break

    acc = acc / total_length * 100.
    mrr = mrr / total_length * 100.

    print('\n*** The total test Acc: ',acc,'  The total test MRR: ',mrr)
    return acc,mrr

def average_status(status):
    acc = []
    mrr = []
    for sta in status:
        acc.append(sta['acc'])
        mrr.append(sta['mrr'])
    aver_acc = np.array(acc).mean()
    aver_mrr = np.array(mrr).mean()
    print('*** The average test Acc: ', aver_acc, '  The average test MRR: ', aver_mrr)
    return aver_acc,aver_mrr

def output_prediction(args,epoch,status,best_imgs,predictions,language):
    root = './result'
    if not os.path.exists(root):
        os.mkdir(root)
    dir_epoch = f'./result/{args.project_name}'
    if not os.path.exists(dir_epoch):
        os.mkdir(dir_epoch)
    dir = f'./result/{args.project_name}/Epoch{epoch+1}'
    if not os.path.exists(dir):
        os.mkdir(dir)

    # 打印预测结果
    out_path = os.path.join(dir,f'Epoch{epoch+1}_{language}_result.txt')
    outfile = open(out_path, 'w', encoding='utf-8')
    outfile.write(f'{language} Accuracy = ' + str(status['acc']) + ' MRR = ' + str(status['mrr']) + '\n')
    for idx, pre in enumerate(best_imgs):
        outfile.write(str(idx + 1) + ' ' + pre + '\n')
    outfile.close()

    prediction_path = os.path.join(dir,f'Epoch{epoch+1}_{language}_prediction.txt')
    prefile = open(prediction_path,'w',encoding='utf-8')
    for k,prediction in enumerate(predictions):
        batch = len(prediction)
        for i in range(batch):
            for j in range(len(prediction[i])):
                prefile.write(prediction[i][j])
                if j < len(prediction[i]) -1:
                    prefile.write('\t')
            if not (k == len(predictions) -1 and i == batch -1):
                prefile.write('\n')
    prefile.close()

def init_logger(args):
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    time_str = str(datetime.now())[:-7].replace(' ', '_').replace(':', '.')
    path = os.path.join(args.log_dir,time_str + f'{args.project_name}')
    if not os.path.exists(path):
        os.mkdir(path)
    log_path = os.path.join(path, f'{args.project_name}_main_log.txt')
    logger = open(log_path, 'w',encoding='utf-8')
    logger.write(str(args) + '\n\n')
    logger.close()

    log_sidetrack_path = os.path.join(path, f'{args.project_name}_sidetrack_log.txt')
    logger_sidetrack = open(log_sidetrack_path, 'w',encoding='utf-8')
    logger_sidetrack.write(str(args) + '\n\n')
    logger_sidetrack.close()
    return log_path,log_sidetrack_path

def update_logger(log_path,epoch,train_status,test_status,total,average,best_epoch):
    logger = open(log_path, 'a+',encoding='utf-8')
    logger.write('>>>>>>>>>>>>>>>>>> (' + str(datetime.now())[:-7] + ') Epoch ' + str(epoch + 1) + ':\n')
    logger.write('------' + 'main pattern 1' + '------' + '\n')
    logger.write('Train Loss: ' + str(train_status['loss']) + '\n')
    logger.write('Learning rate: ' + str(train_status['lr']) + '\n')
    state_name = ['"test_en"','"test_fa"','"test_it"']
    for idx,state in enumerate(test_status):
        logger.write(f'{state_name[idx]} Acc: ' + str(state['acc']) + '    Mrr: ' + str(state['mrr']) + '\n')
    logger.write('\n*** Total Acc: ' + str(total[0]) + '    Mrr: ' + str(total[1]) + '\n')
    logger.write('*** Average Acc: ' + str(average[0]) + '    Mrr: ' + str(average[1]) + '\n')
    logger.write('*** Best epoch @ ' + str(best_epoch) + '\n\n')
    logger.close()

def update_logger_sidetrack(log_path,epoch,train_status,test_status_2,total_2,average_2,test_status_3,total_3,average_3):
    logger = open(log_path, 'a+', encoding='utf-8')
    logger.write('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  Epoch ' + str(epoch + 1) + ':\n')
    logger.write('------' + 'side pattern 2' + '------' + '\n')
    logger.write('Train Loss: ' + str(train_status['loss']) + '\n')
    logger.write('Learning rate: ' + str(train_status['lr']) + '\n')
    state_name = ['"test_en"', '"test_fa"', '"test_it"']
    for idx,state in enumerate(test_status_2):
        logger.write(f'{state_name[idx]} Acc: ' + str(state['acc']) + '    Mrr: ' + str(state['mrr']) + '\n')
    logger.write('\n*** Total Acc: ' + str(total_2[0]) + '    Mrr: ' + str(total_2[1]) + '\n')
    logger.write('*** Average Acc: ' + str(average_2[0]) + '    Mrr: ' + str(average_2[1]) + '\n')

    logger.write('------' + 'side pattern 3' + '------'+ '\n')
    for idx,state in enumerate(test_status_3):
        logger.write(f'{state_name[idx]} Acc: ' + str(state['acc']) + '    Mrr: ' + str(state['mrr']) + '\n')
    logger.write('\n*** Total Acc: ' + str(total_3[0]) + '    Mrr: ' + str(total_3[1]) + '\n')
    logger.write('*** Average Acc: ' + str(average_3[0]) + '    Mrr: ' + str(average_3[1]) + '\n\n')
    logger.close()

def summarize_logger(log_path,best_recorder):
    logger = open(log_path, 'a+')
    logger.write('\n**********' + str(best_recorder) + '\n')
    logger.close()

def pattern(input_pattern):
    if input_pattern == 1:
        print('------'*5,'1. original phrase','------'*5)
    elif input_pattern == 2:
        print('\n')
        print('------' * 5, '2. translated phrase + translated sense', '------' * 5)
    elif input_pattern == 3:
        print('\n')
        print('------' * 5, '3. original phrase + original sense', '------' * 5)
