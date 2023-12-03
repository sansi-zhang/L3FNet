from utils import *
from einops import rearrange
import argparse
from torchvision.transforms import ToTensor


from model.model_Quant import Net_Quant
from model.model_Full import Net_Full



class Test(object):
    def __init__(self, cfg):
        test(cfg)
        
    def test(cfg):
        print("--------------------")
        print("test start!!!")
        
        if cfg.net == 'Net_Quant' or cfg.net == 'Quant':
            net = Net_Quant(cfg)
        elif cfg.net == 'Net_Full' or cfg.net == 'Full':
            net = Net_Full(cfg)
            
        net.to(cfg.device)
        
        
        
        filename_para = cfg.net + '_lr' + str(cfg.lr) + '_n_steps' + str(cfg.n_steps)
        model_path = '../param/' + cfg.net +'/' + filename_para + '_' + str(cfg.n_epochs) + 'best_' + cfg.best +'.pth.tar'
        print('param=', model_path)
      
        model = torch.load(model_path, map_location={'cuda:0': cfg.device})
        net.load_state_dict(model['state_dict'])
        
        scene_list = ['boxes', 'cotton', 'dino', 'sideboard', 'stripes', 'pyramids', 'dots', 'backgammon']
        # scene_list = os.listdir(cfg.testset_dir)

        angRes = cfg.angRes

        for scenes in scene_list:
            print('Working on scene: ' + scenes + '...')
            temp = imageio.v2.imread(cfg.testset_dir + scenes + '/input_Cam000.png')
            lf = np.zeros(shape=(9, 9, temp.shape[0], temp.shape[1], 3), dtype=int)
            for i in range(81):
                temp = imageio.v2.imread(cfg.testset_dir + scenes + '/input_Cam0%.2d.png' % i)
                lf[i // 9, i - 9 * (i // 9), :, :, :] = temp
                
            lf_gray = np.mean((1 / 255) * lf.astype('float32'), axis=-1, keepdims=False)
            angBegin = (9 - angRes) // 2
            lf_angCrop = lf_gray[angBegin:  angBegin + angRes, angBegin: angBegin + angRes, :, :]

            if cfg.crop == False:
                data = rearrange(lf_angCrop, 'u v h w -> (u h) (v w)')
                data = ToTensor()(data.copy())
                data = data.unsqueeze(0)
                with torch.no_grad():
                    disp = net(data.to(cfg.device))
                disp = np.float32(disp[0,0,:,:].data.cpu())

            else:
                patchsize = cfg.patchsize
                stride = patchsize // 2
                data = torch.from_numpy(lf_angCrop)
                sub_lfs = LFdivide(data.unsqueeze(2), patchsize, stride)
                n1, n2, u, v, c, h, w = sub_lfs.shape
                sub_lfs = rearrange(sub_lfs, 'n1 n2 u v c h w -> (n1 n2) u v c h w')
                mini_batch = cfg.minibatch_test
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

            print('Finished! \n')
            mid_dir = cfg.net + '/' + cfg.best
            save_dir = os.path.join(cfg.save_path, mid_dir)
            os.makedirs(save_dir, exist_ok=True) 
            write_pfm(disp, save_dir + '/scenes%s_net%s_Conv3d_mode%s_bit%s_epochs%d.pfm' % (scenes, cfg.net, cfg.Conv3d_mode, cfg.quant_bit, cfg.n_epochs))

        return 






