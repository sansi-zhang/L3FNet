import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from test import Test
from train import Train
from valid import Valid
from utils import *
import argparse

# Settings
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--angRes", type=int, default=7, help="angular resolution")
    parser.add_argument('--model_name', type=str, default='../log/L3FNet')
    parser.add_argument('--testset_dir', type=str, default='../dataset/validation/')
    parser.add_argument('--crop', type=bool, default=True)
    parser.add_argument('--patchsize', type=int, default=48)
    parser.add_argument('--minibatch_test', type=int, default=48)
    parser.add_argument('--model_file', type=str, default='../param/')
    parser.add_argument('--model_path', type=str, default='../param/')
    parser.add_argument('--save_path', type=str, default='../Results/')  
    parser.add_argument('--parallel', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--trainset_dir', type=str, default='../dataset/training/')
    parser.add_argument('--validset_dir', type=str, default='../dataset/validation/')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=7e-4, help='initial learning rate')
    parser.add_argument('--n_epochs', type=int, default=3000, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=700, help='number of epochs to update learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decaying factor')
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--net', type=str, default='quant', help='net mode selecting')
    parser.add_argument('--mode', type=str, default='train', help='train or test mode selecting')
    parser.add_argument('--best', type=str, default='mse', help='param select')
    parser.add_argument('--load_continuetrain', type=bool, default=False, help='continue train')
    
    return parser.parse_args()


def launch(cfg):
    if cfg.mode == 'train':
        Train.train(cfg)
    elif cfg.mode == 'test':
        Test.test(cfg)
    elif cfg.mode == 'whole':
        Train.train(cfg)
        Test.test(cfg)
    elif cfg.mode == 'valid':
        Valid.valid(cfg)
        
        
def main(cfg):
    launch(cfg)

if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
        
        
        
