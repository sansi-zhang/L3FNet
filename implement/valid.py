import time
from datetime import datetime
import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt

from model.model_Quant import Net_Quant
from model.model_Full import Net_Full




class Valid(object):
    def __init__(self, cfg):
        valid(cfg)
        
    def valid(cfg):
        print("--------------------")
        print("valid start!!!")
        if cfg.parallel:
            cfg.device = 'cuda:0'

            
        if cfg.net == 'Net_Quant' or cfg.net == 'Quant':
            net = Net_Quant(cfg)
        elif cfg.net == 'Net_Full' or cfg.net == 'Full':
            net = Net_Full(cfg)

            
            
        net.to(cfg.device)
        cudnn.benchmark = True
        
        
        
        filename_para = cfg.net + '_' + '_lr' + str(cfg.lr) + '_n_steps' + str(cfg.n_steps)
        
        filename =  cfg.net +'/' + filename_para + '_' + str(cfg.n_epochs) + 'best_' + cfg.best +'.pth.tar'
        
        param_path = cfg.model_file + filename
        if os.path.isfile(param_path):
            model = torch.load(param_path, map_location={'cuda:0': cfg.device})
            net.load_state_dict(model['state_dict'], strict=False)
            epoch_state = model["epoch"]
        else:
            print("=> no model found at '{}'".format(cfg.model_file))
            
        Valid.execut(net, cfg)


    def execut(net, cfg):

        scene_list = ['boxes', 'cotton', 'dino', 'sideboard', 'stripes', 'pyramids', 'dots', 'backgammon']
        angRes = cfg.angRes
        totalmse = 0
        filename_txt = (cfg.model_name + cfg.net + '_lr' + str(cfg.lr) + '_n_steps' + str(cfg.n_steps)
                        + '_MSE' + '_epoch' + str(cfg.n_epochs) + '.txt')
        
        txtfile = open(filename_txt, 'a')
        
        txtfile.write('Valid:\t' 'time:{}\t' 'net:{}\t' .format(datetime.now().strftime('%Y%m%d_%H%M%S'), cfg.net))
        txtfile.close()

        for scenes in scene_list:
            lf = np.zeros(shape=(9, 9, 512, 512, 3), dtype=int)
            for i in range(81):
                temp = imageio.v2.imread(cfg.validset_dir + scenes + '/input_Cam0%.2d.png' % i)
                lf[i // 9, i - 9 * (i // 9), :, :, :] = temp
                del temp
            lf = np.mean((1 / 255) * lf.astype('float32'), axis=-1, keepdims=False)
            disp_gt = np.float32(
                read_pfm(cfg.validset_dir + scenes + '/gt_disp_lowres.pfm'))  # load groundtruth disparity map

            angBegin = (9 - angRes) // 2

            lf_angCrop = lf[angBegin:  angBegin + angRes, angBegin: angBegin + angRes, :, :]
            data = torch.from_numpy(lf_angCrop)

            patchsize = 128
            stride = 64
            mini_batch = 8

            sub_lfs = LFdivide(data.unsqueeze(2), patchsize, stride)
            n1, n2, u, v, c, h, w = sub_lfs.shape
            sub_lfs = rearrange(sub_lfs, 'n1 n2 u v c h w -> (n1 n2) u v c h w')

            num_inference = (n1 * n2) // mini_batch
            with torch.no_grad():
                out_disp = []
                for idx_inference in range(num_inference):
                    current_lfs = sub_lfs[idx_inference * mini_batch: (idx_inference + 1) * mini_batch, :, :, :, :, :]
                    input_data = rearrange(current_lfs, 'b u v c h w -> b c (u h) (v w)')
                    # print(input_data.shape)
                    out_disp.append(net(input_data.to(cfg.device)))

                    if (n1 * n2) % mini_batch:
                        current_lfs = sub_lfs[(idx_inference + 1) * mini_batch:, :, :, :, :, :]
                        input_data = rearrange(current_lfs, 'b u v c h w -> b c (u h) (v w)')
                        # print(input_data.shape)
                        out_disp.append(net(input_data.to(cfg.device)))

            out_disps = torch.cat(out_disp, dim=0)
            out_disps = rearrange(out_disps, '(n1 n2) c h w -> n1 n2 c h w', n1=n1, n2=n2)
            disp = LFintegrate(out_disps, patchsize, patchsize // 2)
            disp = disp[0: data.shape[2], 0: data.shape[3]]
            disp = np.float32(disp.data.cpu())

            mse100 = np.mean((disp[11:-11, 11:-11] - disp_gt[11:-11, 11:-11]) ** 2) * 100
            totalmse+=mse100
            txtfile = open(filename_txt, 'a')
            txtfile.write('mse_{}={:3f}\t'.format(scenes, mse100))
            txtfile.close()
        
        txtfile = open(filename_txt, 'a')
        txtfile.write('avg_mse={:3f}\t'.format(totalmse/len(scene_list)))
        txtfile.write('\n')
        txtfile.close()

        return




    

