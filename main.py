from utils import set_seed
import numpy as np
from sklearn import metrics
import math
import copy
from sklearn.model_selection import train_test_split
from config import *
from model import SBResNet, SBViT, SBResNeXt, SBClip, SBViTB,  NFNet_f6, SBConvNext
from dataset import TrainData
from PIL import ImageFile
import torch
import torch.nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
from utils import quadratic_weighted_kappa, write_log
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')

ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_kfold(device, model_origin, dataset, args):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # train every model
    for model_number, (train_index, val_index) in enumerate(kf.split(dataset)):
        write_log(f'Start training model {model_number+1}', args)
        # write_log(f'learning_rate {args.lr} weight_decay{args.weight_decay}', args)
        # train model from scratch
        model = copy.deepcopy(model_origin)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        loss_fn = torch.nn.CrossEntropyLoss()
        best = -math.inf
        best_epoch = 1
        train_fold = torch.utils.data.dataset.Subset(dataset, train_index)
        val_fold = torch.utils.data.dataset.Subset(dataset, val_index)
        train_loader = DataLoader(train_fold,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      drop_last=True,
                                      pin_memory=True)
        val_loader = DataLoader(val_fold,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    drop_last=True,
                                    pin_memory=True)
        train_size = train_loader.__len__()*args.batch_size
        val_size = val_loader.__len__()*args.batch_size
        # train one model for num_epochs epochs
        for epoch in range(1, args.num_epochs):
            
            acc_num = 0
            loss_record = []
            val_auc = []
            model.train()
            pbar = tqdm(train_loader)
            # train through train_loader
            for i, batch in enumerate(pbar):
                img = torch.tensor(batch[0]).to(device)
                gts = torch.tensor(batch[1]).reshape(-1).to(device)

                y = model(img)
                # y_pred = y*2  # N * 1
                with torch.no_grad():
                    result = torch.argmax(y, dim=-1)
                    p = torch.nn.functional.softmax(y, dim=-1)
                    acc_num += (result==gts).sum().item()

                y = y.to(torch.float)
                gts = gts.to(torch.long)
                loss = loss_fn(input=y.reshape(-1, 3), target=gts.reshape(-1))
                loss_record.append(loss.item() * y.size(0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_description(f"loss: {loss:.5f}")
            mean_train_loss = sum(loss_record) / train_size
            train_acc = acc_num / train_size
            write_log('Epoch{}: training accuracy:{:.4f}, training loss:{:.4f}'.format(
                epoch, train_acc, mean_train_loss), args=args)
            # evaluate model on validation set
            model.eval()
            acc_num = 0
            loss_record = []
            pbar = tqdm(val_loader)
            p_list = []
            gts_list = []
            gt_list = []
            pred_list = []
            for i, batch in enumerate(pbar):
                img = torch.tensor(batch[0]).to(device)
            # print(torch.tensor(batch[1]))
                gts = torch.tensor(batch[1]).reshape(-1).to(device)
                with torch.no_grad():
                    y = model(img)
                    result = torch.argmax(y, dim=-1)
                    acc_num += (result==gts).sum().item()
                    y= y.to(torch.float)
                    gts= gts.to(torch.long)
                    loss = loss_fn(input=y.reshape(-1, 3), target=gts.reshape(-1))
                    p = torch.nn.functional.softmax(y, dim=-1)
                    gts_list.append(gts.cpu().numpy())
                    p_list.append(p.cpu().numpy())
                    gt_list.append(gts)
                    pred_list.append(result)


                
                # print(p)
            
                loss_record.append(loss.item()*y.size(0))
            mean_valid_loss = sum(loss_record) / val_size
            all_target = np.concatenate(gts_list)
            all_pred = np.concatenate(p_list)
            all_gt = torch.cat(gt_list)
            all_p = torch.cat(pred_list)

            valid_auc = metrics.roc_auc_score(
                all_target, all_pred, multi_class='ovo')
            kappa = quadratic_weighted_kappa(
                np.array(all_gt.cpu()), np.array(all_p.cpu()))

            valid_acc = acc_num / val_size
            if kappa + valid_acc>best:
                best = valid_acc+kappa
                best_epoch = epoch
                if not os.path.exists('saved_models/k_fold'):
                    os.mkdir('saved_models/k_fold')
                if args.save_model:
                    torch.save(model,
                            f'./saved_models/k_fold/MODEL{model_number+1}_{args.exp_name}_{args.task}_{args.model}_ckpt_{epoch}_with_acc_{best:.4f}_kappa_{kappa}.pth')

            write_log(
                'Epoch{}: valid accuracy:{:.4f}, auc :{:.4f}, kappa :{:.4f}'.format(epoch, valid_acc, valid_auc, kappa), args=args)
        write_log(
            f'Model {model_number+1} finished training, with best epoch {best_epoch} and best acc {best:.4f}', args=args)


def train(device, logger, model, train_loader, val_loader, train_size, val_size, num_epochs, lr, weight_decay, args):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, num_epochs, lr / 1000)
    loss_fn = torch.nn.CrossEntropyLoss()
    best = -math.inf
    best_epoch = 1
    for ep in range(1, num_epochs):

        acc_num = 0
        loss_record = []
        val_auc = []
        model.train()
        pbar = tqdm(train_loader)
        for i, batch in enumerate(pbar):
            img = torch.tensor(batch[0]).to(device)
            gts = torch.tensor(batch[1]).reshape(-1).to(device)

            y = model(img)

            with torch.no_grad():
                result = torch.argmax(y, dim=-1)
                p = torch.nn.functional.softmax(y, dim=-1)
                acc_num += (result == gts).sum().item()

            y = y.to(torch.float)
            gts = gts.to(torch.long)
            loss = loss_fn(input=y.reshape(-1, 3), target=gts.reshape(-1))

            loss_record.append(loss.item() * y.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(f"loss: {loss:.5f}")
        lr_scheduler.step()
        # torch.save(model, f'./ckpt_{ep}.pth')
        mean_train_loss = sum(loss_record) / train_size
        train_acc = acc_num / train_size
        write_log('Epoch{}: training accuracy:{:.4f}, training loss:{:.4f}'.format(
            ep, train_acc, mean_train_loss), args=args)
        model.eval()
        acc_num = 0
        loss_record = []
        pbar = tqdm(val_loader)
        p_list = []
        gts_list = []
        gt_list = []
        pred_list = []
        for i, batch in enumerate(pbar):
            img = torch.tensor(batch[0]).to(device)

            gts = torch.tensor(batch[1]).reshape(-1).to(device)
            with torch.no_grad():
                y = model(img)
                result = torch.argmax(y, dim=-1)
                acc_num += (result == gts).sum().item()
                y = y.to(torch.float)
                gts = gts.to(torch.long)
                loss = loss_fn(input=y.reshape(-1, 3), target=gts.reshape(-1))
                p = torch.nn.functional.softmax(y, dim=-1)
                gts_list.append(gts.cpu().numpy())
                p_list.append(p.cpu().numpy())
                gt_list.append(gts)
                pred_list.append(result)

            loss_record.append(loss.item() * y.size(0))
        mean_valid_loss = sum(loss_record) / val_size
        all_target = np.concatenate(gts_list)
        all_pred = np.concatenate(p_list)
        all_gt = torch.cat(gt_list)
        all_p = torch.cat(pred_list)

        valid_auc = metrics.roc_auc_score(
            all_target, all_pred, multi_class='ovo')
        kappa = quadratic_weighted_kappa(
            np.array(all_gt.cpu()), np.array(all_p.cpu()))

        valid_acc = acc_num / val_size
        if  kappa +valid_acc>best:
            best = valid_acc+kappa
            best_epoch = ep
            if not os.path.exists('saved_models/args.model'):
                os.mkdir('saved_models/args.model')
            if args.save_model:
                torch.save(model,
                           f'./saved_models/args.model/{args.exp_name}_{args.task}_{args.model}_ckpt_{ep}_with_auc_{best:.4f}_{args.device}_{args.transform}.pth')

        write_log(
            'Epoch{}: valid accuracy:{:.4f}, auc :{:.4f}, kappa :{:.4f}'.format(ep, valid_acc, valid_auc, kappa), args=args)
    return best_epoch, best


if __name__ == '__main__':
    set_seed(42)
    args, logger = get_config()
    args.device = 'cpu' if args.device < 0 else 'cuda:%i' % args.device
    args.device = torch.device(args.device)
    model_dict = {
        'resnet': SBResNet,
        'resnext': SBResNeXt,
        'vit': SBViT,
        'clip': SBClip,
        'vitb': SBViTB,
        'nfnet': NFNet_f6,
        'convnext': SBConvNext
    }
    write_log(args.device, args)
    model = model_dict[args.model]().to(args.device)
    img_size = 512
    if args.model == 'vitb':
        img_size = 384
        
    if args.model == 'clip':
        img_size = 224
    
    if args.task == 'tsk2':
        dataset = TrainData('./data', image_size=img_size, transform=args.transform)
    else:
        dataset = TrainData('../sb/data3', image_size=512,
                            transform=args.transform)
    if args.k_fold:
        train_kfold(
            device=args.device,
            model_origin=model,
            dataset=dataset,
            args=args
        )
    else:
        # get dataset
        train_data, val_data = train_test_split(
            dataset, test_size=0.4, train_size=0.6, random_state=args.seed)
        train_loader = DataLoader(train_data,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  pin_memory=True)
        valid_loader = DataLoader(val_data,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  pin_memory=True)

        best_epoch, best_kappa = train(device=args.device, logger=logger, model=model, train_loader=train_loader,
                                       val_loader=valid_loader, train_size=len(train_data), val_size=len(val_data),
                                       num_epochs=args.num_epochs, lr=args.lr, weight_decay=args.weight_decay, args=args)
        write_log('Best accuracy:{:.4f}'.format(best_kappa), args=args)
