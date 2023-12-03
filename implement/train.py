import time
from datetime import datetime
import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt

from model.model_Quant2 import Net_Quant2
from model.model_Quant import Net_Quant
from model.model_Full import Net_Full
from model.model_Full2 import Net_Full2
from model.model_w8bit import Net_w8bit
from model.model_w6bit import Net_w6bit
from model.model_w2bit import Net_w2bit
from model.model_prune import Net_prune
# from model.model_w6bit import Net_w6bit


class Train(object):
    def __init__(self, cfg):
        train(cfg)
        
    def save_ckpt(state, save_path='./param/', subdirectory='', filename='checkpoint.pth.tar'):
        save_dir = os.path.join(save_path, subdirectory)
        os.makedirs(save_dir, exist_ok=True)  
        torch.save(state, os.path.join(save_dir, filename))
        
    def train(cfg):
        print("--------------------")
        print("train start!!!")
        if cfg.parallel:
            cfg.device = 'cuda:0'

        if cfg.net == 'Net_Quant' or cfg.net == 'Quant':
            net = Net_Quant(cfg)
        elif cfg.net == 'Net_Full' or cfg.net == 'Full':
            net = Net_Full(cfg)
        elif cfg.net == 'Net_Full2':
            net = Net_Full2(cfg)
        elif cfg.net == 'Net_Quant2':
            net = Net_Quant2(cfg)
        elif cfg.net == 'Net_w8bit':
            net = Net_w8bit(cfg)
        elif cfg.net == 'Net_w6bit':
            net = Net_w6bit(cfg)
        elif cfg.net == 'Net_w2bit':
            net = Net_w2bit(cfg)
        elif cfg.net == 'Net_prune':
            net = Net_prune(cfg)
    
        net.to(cfg.device)
        cudnn.benchmark = True
        epoch_state = 0
        min_loss = float('inf')
        min_mse = min_mse_loss = float('inf')
        min_loss_epoch = min_mse_epoch = 0
        
        
        if cfg.load_continuetrain:
            filename_para = ('../param/' + cfg.net + '/' + cfg.net + '_lr' + str(cfg.lr) + '_n_steps' + str(cfg.n_steps) + '_' + str(cfg.n_epochs) + 'best_mse' + '.pth.tar')
            
            if os.path.isfile(filename_para):
                model = torch.load(filename_para, map_location={'cuda:0': cfg.device})
                net.load_state_dict(model['state_dict'], strict=False)
                epoch_state = model["epoch"]
                print("=> model found at '{}'".format(filename_para))
            else:
                print("=> no model found at '{}'".format(filename_para))
        
        if cfg.load_pretrain:
            if os.path.isfile(cfg.model_path):
                model = torch.load(cfg.model_path, map_location={'cuda:0': cfg.device})
                net.load_state_dict(model['state_dict'], strict=False)
                # epoch_state = model["epoch"]
                print("=> model found at '{}'".format(cfg.model_path))
            else:
                print("=> no model found at '{}'".format(cfg.load_model))

        if cfg.parallel:
            net = torch.nn.DataParallel(net, device_ids=[0, 1])
            
            
        criterion_Loss = torch.nn.L1Loss().to(cfg.device)
        optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)
        scheduler._step_count = epoch_state

        loss_list = []
        epoch_history = []

        for idx_epoch in range(epoch_state, cfg.n_epochs):
            train_set = TrainSetLoader(cfg)
            train_loader = DataLoader(dataset=train_set, num_workers=cfg.num_workers, batch_size=cfg.batch_size, shuffle=True)
            loss_epoch = []
            for idx_iter, (data, dispGT) in tqdm(enumerate(train_loader), total=len(train_loader)):           
                data, dispGT = Variable(data).to(cfg.device), Variable(dispGT).to(cfg.device)
                disp = net(data)

                loss = criterion_Loss(disp, dispGT[:, 0, :, :].unsqueeze(1))         #disp:torch.Size([4, 1, 48, 48])dispGT:torch.Size([4, 2, 48, 48])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_epoch.append(loss.data.cpu())
                
            
            filename_para = cfg.net + '_lr' + str(cfg.lr) + '_n_steps' + str(cfg.n_steps) 
                
            
            if idx_epoch % 1 == 0:         
                
                print(time.ctime()[4:-5] + ' Epoch----%5d, loss---%f' % (idx_epoch + 1, 
                                                                        float(np.array(loss_epoch).mean())))
                
                
                if cfg.parallel:
                    Train.save_ckpt({
                    'epoch': idx_epoch + 1,
                    'state_dict': net.module.state_dict(),
                }, save_path='../param/', subdirectory=cfg.net, filename=filename_para + '_' + str(cfg.n_epochs) + '.pth.tar')
                else:
                    Train.save_ckpt({
                        'epoch': idx_epoch + 1,
                        'state_dict': net.state_dict(),
                    }, save_path='../param/', subdirectory=cfg.net, filename=filename_para + '_' + str(cfg.n_epochs) + '.pth.tar')
                
            if np.array(loss_epoch).mean() < min_loss:
                min_loss = np.array(loss_epoch).mean()
                min_loss_epoch = idx_epoch
                
                Train.save_ckpt({
                    'epoch': idx_epoch + 1,
                    'state_dict': net.state_dict(),
                }, save_path='../param/', subdirectory=cfg.net, filename=filename_para + '_' + str(cfg.n_epochs) + 'best_loss' + '.pth.tar')
                                       
                      
            if idx_epoch % 20 == 19:
                loss_list.append(float(np.array(loss_epoch).mean()))
                epoch_history.append(idx_epoch)
                plt.figure()
                plt.title('loss')
                plt.plot(epoch_history, loss_list, label='Training loss')  
                plt.legend()  
                plt.savefig('../loss/' + filename_para + '.jpg')
                plt.close()
                
                loss_now = np.array(loss_epoch).mean()
                avg_mse = Train.valid(net, cfg, idx_epoch + 1, loss_now, min_loss)
                if avg_mse < min_mse:
                    min_mse = avg_mse
                    min_mse_epoch = idx_epoch
                    min_mse_loss  = np.array(loss_epoch).mean()
                    Train.save_ckpt({
                        'epoch': idx_epoch + 1,
                        'state_dict': net.state_dict(),
                    }, save_path='../param/', subdirectory=cfg.net, filename=filename_para + '_' + str(cfg.n_epochs) + 'best_mse' + '.pth.tar')
            
            if(idx_epoch==cfg.n_epochs-1):
                filename_txt = (cfg.model_name + cfg.net + '_lr' + str(cfg.lr) + '_n_steps' + str(cfg.n_steps)
                        + '_MSE' + '_epoch' + str(cfg.n_epochs) + '.txt')
        
                txtfile = open(filename_txt, 'a')
                txtfile.write('Best:\t' 'best_loss:epoch={}, min_loss={}\t' 
                              'best_mse:epoch={}, min_mse_loss={}, min_mse={}\t'.format(min_loss_epoch, min_loss, min_mse_epoch, min_mse_loss, min_mse))
                txtfile.close()
            
            scheduler.step()


    def valid(net, cfg, epoch, loss_now, min_loss):

        scene_list = ['boxes', 'cotton', 'dino', 'sideboard', 'stripes', 'pyramids', 'dots', 'backgammon']
        angRes = cfg.angRes
        totalmse = 0
        
        filename_txt = (cfg.model_name + cfg.net + '_lr' + str(cfg.lr) + '_n_steps' + str(cfg.n_steps) + '_MSE' + '_epoch' + str(cfg.n_epochs) + '.txt')
        
        txtfile = open(filename_txt, 'a')
        
        txtfile.write('Epoch={}:\t' 'time:{}\t' 'net:{}\t' 'loss_now:{}\t' 'loss_min:{}\t'.format(epoch, datetime.now().strftime('%Y%m%d_%H%M%S'), cfg.net, loss_now, min_loss))
        
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
                    out_disp.append(net(input_data.to(cfg.device)))

                    if (n1 * n2) % mini_batch:
                        current_lfs = sub_lfs[(idx_inference + 1) * mini_batch:, :, :, :, :, :]
                        input_data = rearrange(current_lfs, 'b u v c h w -> b c (u h) (v w)')
                        out_disp.append(net(input_data.to(cfg.device)))

            out_disps = torch.cat(out_disp, dim=0)
            out_disps = rearrange(out_disps, '(n1 n2) c h w -> n1 n2 c h w', n1=n1, n2=n2)
            disp = LFintegrate(out_disps, patchsize, patchsize // 2)
            disp = disp[0: data.shape[2], 0: data.shape[3]]
            disp = np.float32(disp.data.cpu())

            mse100 = np.mean((disp[11:-11, 11:-11] - disp_gt[11:-11, 11:-11]) ** 2) * 100
            totalmse += mse100
            txtfile = open(filename_txt, 'a')
            txtfile.write('mse_{}={:3f}\t'.format(scenes, mse100))
            txtfile.close()

        txtfile = open(filename_txt, 'a')
        txtfile.write('avg_mse={:3f}\t'.format(totalmse/len(scene_list)))
        txtfile.write('\n')
        txtfile.close()

        return  totalmse/len(scene_list)




    

