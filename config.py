import os
import sys
import logging
import argparse
from datetime import datetime


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--log',type=str,default='./logs')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--aug', action='store_true')
    parser.add_argument('--model_path', default='./saved_models')
    parser.add_argument('--task', type=str, default='tsk2')
    parser.add_argument('--model', type=str, default='convnext')
    parser.add_argument('--exp_name', type=str, default='base')
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--transform', type=bool, default=True)
    parser.add_argument('--k_fold',action='store_true')
    args = parser.parse_args()

    args.log_name = '{}-{}-{}-{}-{}-{}.log'.format(args.task, args.model, args.batch_size, args.lr, args.weight_decay,
                                                   datetime.now().strftime('%Y-%m-%d_%H-%M-%S')[2:])
    if not os.path.exists('logs'):
        os.mkdir('logs')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.addHandler(logging.FileHandler(os.path.join('logs', args.log_name)))
    return args, logger
