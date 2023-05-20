import argparse
import os
import yaml
from utils import seed_everything,cosine_lr_schedule,frozen_model,display_modelpara,calculate_modelpara
from utils import integrate_test_status,average_status,output_prediction,init_logger,update_logger,summarize_logger,pattern,update_logger_sidetrack
from load_datasets import load_data
import torch
from models.mta_model import mtam
from train import train_fn
from evaluate import evaluation_fn
from torch.optim import AdamW

def set_config():
    parser = argparse.ArgumentParser(description="Finetune on T-VWSD")
    parser.add_argument('--project_name',type=str,required=False, default='open12layer',help='a name of this project')
    parser.add_argument('--config', default='./vwsd.yaml')
    parser.add_argument('--device', type=int, required=False, default=0)

    parser.add_argument('--lr', type=float, required=False, default=1e-4, help="learning rate")
    parser.add_argument('--min_lr', type=float, required=False, default=0.0, help="minimum learning rate")
    parser.add_argument('--weight_decay', type=float, required=False, default=0.05, help="weight_decay")
    parser.add_argument('--patience', type=int, required=False, default=6, help="patience for rlp")

    parser.add_argument('--use_checkpoint', action='store_true', default=False, help="use pretrained weights or not")
    parser.add_argument('--evaluate', action='store_true', default=False, help="evaluate or not")

    parser.add_argument('--open', type=int, required=False, default=12)

    args = parser.parse_args()

    config = yaml.load(open('vwsd.yaml', 'r', encoding='utf-8'), Loader=yaml.Loader)
    args.L_VWSD_dir = config['L_VWSD_dir']
    args.official_data_dir = config['official_data_dir']
    args.train_batch_size = config['train_batch_size']
    args.eval_batch_size = config['eval_batch_size']
    args.num_workers = config['num_workers']
    args.epochs = config['epochs']
    args.alpha = config['alpha']
    args.save_dir = config['save_dir']
    args.log_dir = config['log_dir']
    args.seed = config['seed']
    return args


if __name__ == '__main__':
    args = set_config()
    seed_everything(args.seed)

    # 1. get data
    data = load_data(args)

    model = mtam(args,data)
    if args.use_checkpoint:
        model.load_state_dict(torch.load(os.path.join(args.save_dir, 'checkpoint.pt')))
        print('>>>>>>Using the model checkpoint :\n')
    model.to(args.device)
    frozen_model(model, mode='partial',open=args.open)
    # display_modelpara(model)
    calculate_modelpara(model)
    # exit()

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()

    # init logger before training
    if not args.evaluate:
        log_path,log_sidetrack_path = init_logger(args)
    # set the recorder for saving the best model
    recorder = {
        'best_epoch': 0,
        'best_en_acc': 0,
        'best_en_mrr': 0,
        'best_fa_acc': 0,
        'best_fa_mrr': 0,
        'best_it_acc': 0,
        'best_it_mrr': 0,
        'best_total_acc': 0,
        'best_total_mrr': 0
    }

    # 3.train and evaluation
    for epoch in range(args.epochs):
        print('======' * 8, f'Epoch {str(epoch + 1)}', '======' * 8)
        cosine_lr_schedule(optimizer, epoch, args.patience, args.lr, args.min_lr)
        if not args.evaluate:
            train_status = train_fn(epoch, args, data, model, optimizer,scaler, mode='train')

        ### main pattern
        pattern(1)
        en_status, en_best_imgs, en_SORT10, en_GOLD = evaluation_fn(epoch, args, data, model, mode='test_en',input_pattern=1)
        fa_status, fa_best_imgs, fa_SORT10, fa_GOLD = evaluation_fn(epoch, args, data, model, mode='test_fa',input_pattern=1)
        it_status, it_best_imgs, it_SORT10, it_GOLD = evaluation_fn(epoch, args, data, model, mode='test_it',input_pattern=1)

        # calculate total metrics
        total_acc, total_mrr = integrate_test_status([en_SORT10, fa_SORT10, it_SORT10], [en_GOLD, fa_GOLD, it_GOLD])
        # calculate average metrics
        aver_acc,aver_mrr = average_status([en_status,fa_status,it_status])


        ############ evaluation sidetrack
        pattern(2)
        en_status_2, en_best_imgs_2, en_SORT10_2, en_GOLD_2 = evaluation_fn(epoch, args, data, model, mode='test_en',input_pattern=2)
        fa_status_2, fa_best_imgs_2, fa_SORT10_2, fa_GOLD_2 = evaluation_fn(epoch, args, data, model, mode='test_fa',input_pattern=2)
        it_status_2, it_best_imgs_2, it_SORT10_2, it_GOLD_2 = evaluation_fn(epoch, args, data, model, mode='test_it',input_pattern=2)
        # calculate total metrics
        total_acc_2, total_mrr_2 = integrate_test_status([en_SORT10_2, fa_SORT10_2, it_SORT10_2], [en_GOLD_2, fa_GOLD_2, it_GOLD_2])
        # calculate average metrics
        aver_acc_2, aver_mrr_2 = average_status([en_status_2, fa_status_2, it_status_2])

        pattern(3)
        en_status_3, en_best_imgs_3, en_SORT10_3, en_GOLD_3 = evaluation_fn(epoch, args, data, model, mode='test_en',input_pattern=3)
        fa_status_3, fa_best_imgs_3, fa_SORT10_3, fa_GOLD_3 = evaluation_fn(epoch, args, data, model, mode='test_fa',input_pattern=3)
        it_status_3, it_best_imgs_3, it_SORT10_3, it_GOLD_3 = evaluation_fn(epoch, args, data, model, mode='test_it',input_pattern=3)
        # calculate total metrics
        total_acc_3, total_mrr_3 = integrate_test_status([en_SORT10_3, fa_SORT10_3, it_SORT10_3],[en_GOLD_3, fa_GOLD_3, it_GOLD_3])
        # calculate average metrics
        aver_acc_3, aver_mrr_3 = average_status([en_status_3, fa_status_3, it_status_3])

        if args.evaluate:
            break

        # output prediction.txt
        output_prediction(args,epoch, en_status, en_best_imgs, en_SORT10, language='en')
        output_prediction(args,epoch, fa_status, fa_best_imgs, fa_SORT10, language='fa')
        output_prediction(args,epoch, it_status, it_best_imgs, it_SORT10, language='it')

        ### save model
        if aver_acc > recorder['best_total_acc']:
            if not os.path.exists(args.save_dir):
                os.mkdir(args.save_dir)
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'checkpoint.pt'))
            recorder['best_epoch'] = epoch + 1
            recorder['best_en_acc'] = en_status['acc']
            recorder['best_en_mrr'] = en_status['mrr']
            recorder['best_fa_acc'] = fa_status['acc']
            recorder['best_fa_mrr'] = fa_status['mrr']
            recorder['best_it_acc'] = it_status['acc']
            recorder['best_it_mrr'] = it_status['mrr']
            recorder['best_total_acc'] = aver_acc
            recorder['best_total_mrr'] = aver_mrr
        elif aver_acc == recorder['best_total_acc'] and aver_mrr > recorder['best_total_mrr']:
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'checkpoint.pt'))
            recorder['best_epoch'] = epoch + 1
            recorder['best_en_acc'] = en_status['acc']
            recorder['best_en_mrr'] = en_status['mrr']
            recorder['best_fa_acc'] = fa_status['acc']
            recorder['best_fa_mrr'] = fa_status['mrr']
            recorder['best_it_acc'] = it_status['acc']
            recorder['best_it_mrr'] = it_status['mrr']
            recorder['best_total_acc'] = aver_acc
            recorder['best_total_mrr'] = aver_mrr

        best_epoch = recorder['best_epoch']
        print(f'best epoch @ {best_epoch}')
        # update logger
        update_logger(log_path, epoch, train_status, [en_status, fa_status, it_status], [total_acc, total_mrr],[aver_acc, aver_mrr],best_epoch)

        # update logger sidetrack
        update_logger_sidetrack(log_sidetrack_path, epoch,train_status,
                                [en_status_2, fa_status_2, it_status_2],
                                [total_acc_2, total_mrr_2],
                                [aver_acc_2, aver_mrr_2],
                                [en_status_3, fa_status_3, it_status_3],
                                [total_acc_3, total_mrr_3],
                                [aver_acc_3, aver_mrr_3]
                                )

    if not args.evaluate:
        best_recorder = f'At the end, the best evaluating description: {recorder}'
        print('\n'+best_recorder)
        summarize_logger(log_path, best_recorder)