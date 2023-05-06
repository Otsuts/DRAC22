import pandas as pd
from config import *
from dataset import TestData
from PIL import ImageFile
import torch
import torch.nn
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings('ignore')

ImageFile.LOAD_TRUNCATED_IMAGES = True


def test(test_loader, model_name):
    prediction, p1, p2, p3 = [], [], [], []
    model = torch.load(model_name).to('cuda:2')
    print(f"model loaded from {model_name}")
    for i in test_loader:
        X = torch.tensor(i).to('cuda:2')
        num = X.shape[0]
        y = model(X)
        if y.shape[1] == 3:
            result = torch.argmax(y, dim=-1)
            p = torch.nn.functional.softmax(y, dim=-1)
        else:
            y_pred = y*2  # N * 1
            result = y_pred.round().squeeze().long()
            p = torch.zeros(X.shape[0], 3).long()
            p = p.scatter_(dim=1, index=result.unsqueeze(
                1).cpu(), src=torch.ones(X.shape[0], 3).long()).float()
        for j in range(0, num):
            prediction.append(result[j].item())
            p1.append(p[j][0].item())
            p2.append(p[j][1].item())
            p3.append(p[j][2].item())

    return prediction, p1, p2, p3


def test_ensemble(test_loader, model_name_list):
    _, prob1, prob2, prob3 = test(test_loader, model_name_list[0])
    model_number = len(model_name_list)
    for model_name in model_name_list[1:]:
        _, p1, p2, p3 = test(test_loader, model_name)
        for i in range(len(prob1)):
            prob1[i] += p1[i]
            prob2[i] += p2[i]
            prob3[i] += p3[i]
    prediction = []
    for i in range(len(prob1)):
        prob1[i] /= model_number
        prob2[i] /= model_number
        prob3[i] /= model_number
        tmp = max([prob1[i], prob2[i], prob3[i]])
        prediction.append([prob1[i], prob2[i], prob3[i]].index(tmp))

    return prediction, prob1, prob2, prob3


if __name__ == '__main__':
    args, logger = get_config()
    model_name = 'saved_models/replay_tsk2_resnet_ckpt_40_with_auc_0.8877_cuda:5_True.pth'
    if args.task == 'tsk2':
        dataset = TestData('./data', image_size=512)
    else:
        dataset = TestData('./data3', image_size=512)
    dataset = TestData('./data3', image_size=384)
    name = dataset.names()
    test_loader = DataLoader(dataset,
                             batch_size=2,
                             shuffle=False,
                             drop_last=True,
                             pin_memory=True)

    model_name_list = [
        'saved_models/k_fold/MODEL1_base_tsk3_vitb_ckpt_52_with_acc_0.8583_kappa_0.8593103448275862.pth',
        'saved_models/k_fold/MODEL2_base_tsk3_vitb_ckpt_47_with_acc_0.8500_kappa_0.8519939701247088.pth',
        'saved_models/k_fold/MODEL3_base_tsk3_vitb_ckpt_75_with_acc_0.8667_kappa_0.8434442270058709.pth',
        'saved_models/k_fold/MODEL4_base_tsk3_vitb_ckpt_12_with_acc_0.8417_kappa_0.8559514783927218.pth',
        'saved_models/k_fold/MODEL5_base_tsk3_vitb_ckpt_87_with_acc_0.8167_kappa_0.75.pth'

    ]
    # pre, p1, p2, p3 = test(
    #     test_loader=test_loader, model_name=model_name)
    pre, p1, p2, p3 = test_ensemble(
        test_loader=test_loader, model_name_list=model_name_list)
    list_res = []
    for i in range(len(name)):
        list_res.append([name[i], pre[i], p1[i], p2[i], p3[i]])
    column_name = ['case', 'class', 'P0', 'P1', 'P2']
    csv_name = f'./submit/tsk3.csv'
    df = pd.DataFrame(list_res, columns=column_name)
    df.to_csv(csv_name, index=None)
