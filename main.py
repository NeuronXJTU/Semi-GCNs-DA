import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
import random
import shutil
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import warnings
import train

def main(args):
    '''
    Main
    '''
    # This is not recommended:
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device is', args.device)
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    train.train(args=args)

def parse_args():
    parser = argparse.ArgumentParser(description='Recursion')
    # Basic settings
    parser.add_argument('--seed', type=int, default=233)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=16)
    # Train settings
    parser.add_argument('--pre_epochs', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=50)
    # Optimizer
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--begain_ent', type=int, default=1)
    # Txts
    parser.add_argument('--txt_dir', type=str, default=''/media/oem/sda21/zk/DATA/ultrasonic'')
    parser.add_argument('--source_txt', type=str, default='./txt/0/source.txt')
    parser.add_argument('--target_unlabeled_txt', type=str, default='./txt/0/targetUnlabel.txt')
    parser.add_argument('--target_test_txt', type=str, default='./txt/0/targetTest.txt')
    parser.add_argument('--target_labeled_txt', type=str, default='./txt/0/targetReal.txt')
    # Directory for output
    parser.add_argument('--out_dir', type=str, default='./')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    args = parse_args()
    main(args=args)
