import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import GANLoss, ReconstructionLoss, SSIMLoss
from models.modules.Quantization import Quantization
from models.modules.Jpeg_Compress import Jpeg_Compress_Layer
from models.modules.DiffJPEG.DiffJPEG import DiffJPEG
import lpips
import utils.util as util
import cv2
import numpy as np
logger = logging.getLogger('base')

class IRNpModel(BaseModel):
    def __init__(self, opt):
        super(IRNpModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        test_opt = opt['test']
        self.train_opt = train_opt
        self.test_opt = test_opt
        self.prompt = None

        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        self.Quantization = Quantization()
        if opt['compress_mode'] == 'diffjpeg':
            print("Using diffjpeg compression")
            r = opt['datasets']['train']['resolution']
            self.Compression = DiffJPEG(r, r, quality=95).to(self.device)
        else:
            print("Using regular jpeg compression")
            self.Compression = Jpeg_Compress_Layer()

        if self.is_train:
            # self.netD = networks.define_D(opt).to(self.device)
            # if opt['dist']:
            #     self.netD = DistributedDataParallel(self.netD, device_ids=[torch.cuda.current_device()])
            # else:
            #     self.netD = DataParallel(self.netD)

            self.netG.train()
            # self.netD.train()

            # loss
            self.Reconstruction_forw = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_forw'])
            self.Reconstruction_back = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_back'])
            self.LPIPS_loss = lpips.LPIPS(net=train_opt['lpips_net']).to(self.device)
            self.ssim_loss = SSIMLoss().to(self.device)
            # feature loss
            # if train_opt['feature_weight'] > 0:
            #     self.Reconstructionf = ReconstructionLoss(losstype=self.train_opt['feature_criterion'])

            #     self.l_fea_w = train_opt['feature_weight']
            #     self.netF = networks.define_F(opt, use_bn=False).to(self.device)
            #     if opt['dist']:
            #         self.netF = DistributedDataParallel(self.netF, device_ids=[torch.cuda.current_device()])
            #     else:
            #         self.netF = DataParallel(self.netF)
            # else:
            #     self.l_fea_w = 0

            # GD gan loss
            # self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
            # self.l_gan_w = train_opt['gan_weight']

            # D_update_ratio and D_init_iters
            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0


            # optimizers
            # G
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            # D
            # wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            # self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=train_opt['lr_D'], weight_decay=wd_D, betas=(train_opt['beta1_D'], train_opt['beta2_D']))
            # self.optimizers.append(self.optimizer_D)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def feed_data(self, data, identity = False):
        self.ref_L = data['LQ'].to(self.device)  # LQ
        self.real_H = data['GT'].to(self.device)  # GT
        if data.get('prompt', None) is not None:
            self.prompt = data['prompt']
        else:
            self.prompt = None
        
        if identity:
            self.uninv_input = self.ref_L.clone()
        else:
            self.uninv_input = None

    def gaussian_batch(self, dims):
        return torch.randn(tuple(dims)).to(self.device)

    # def loss_forward(self, out, y):
    #     l_forw_fit = self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out[:, :3, :, :], y)

    #     return l_forw_fit

    def loss_forward(self, out, y, z = None):
        losses = {}
        l_forw_fit = self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out[:, :3, :, :], y)
        if self.train_opt['lambda_fit_forw'] > 0:
            losses['l_forw_fit'] = l_forw_fit
        # LPIPS loss
        if self.train_opt['lambda_lpips_forw'] > 0:
            l_forw_lpips = self.train_opt['lambda_lpips_forw'] * self.LPIPS_loss(out[:, :3, :, :], y).mean()
            losses['l_forw_lpips'] = l_forw_lpips

        if z is not None:
            z = z.reshape([out.shape[0], -1])
            l_forw_ce = self.train_opt['lambda_ce_forw'] * torch.sum(z**2) / z.shape[0]
            losses['l_forw_ce'] = l_forw_ce

        if self.train_opt['lambda_ssim_forw'] > 0:
            l_forw_ssim = self.train_opt['lambda_ssim_forw'] * self.ssim_loss(out[:, :3, :, :], y)
            losses['l_forw_ssim'] = l_forw_ssim

        return losses

    def loss_backward(self, x, x_samples):
        losses = {}
        x_samples_image = x_samples[:, :3, :, :]
        l_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(x, x_samples_image)
        if self.train_opt['lambda_rec_back'] > 0:
            losses['l_back_rec'] = l_back_rec
        # LPIPS loss
        if self.train_opt['lambda_lpips_back'] > 0:
            l_back_lpips = self.train_opt['lambda_lpips_back'] * self.LPIPS_loss(x, x_samples_image).mean()
            losses['l_back_lpips'] = l_back_lpips

        # SSIM loss
        if self.train_opt['lambda_ssim_back'] > 0:
            l_back_ssim = self.train_opt['lambda_ssim_back'] * self.ssim_loss(x, x_samples_image)
            losses['l_back_ssim'] = l_back_ssim
        # feature loss
        # if self.l_fea_w > 0:
        #     l_back_fea = self.feature_loss(x, x_samples_image)
        # else:
        #     l_back_fea = torch.tensor(0)

        # GAN loss
        # pred_g_fake = self.netD(x_samples_image)
        # if self.opt['train']['gan_type'] == 'gan':
        #     l_back_gan = self.l_gan_w * self.cri_gan(pred_g_fake, True)
        # elif self.opt['train']['gan_type'] == 'ragan':
        #     pred_d_real = self.netD(x).detach()
        #     l_back_gan = self.l_gan_w * (self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) + self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2

        return losses

    def feature_loss(self, real, fake):
        real_fea = self.netF(real).detach()
        fake_fea = self.netF(fake)
        l_g_fea = self.l_fea_w * self.Reconstructionf(real_fea, fake_fea)
        
        return l_g_fea
        

    def optimize_parameters(self, step, compress_aware = False):
        # G
        # for p in self.netD.parameters():
        #     p.requires_grad = False

        self.optimizer_G.zero_grad()

        self.input = self.real_H
        if self.prompt is not None:
            self.output = self.netG(x=self.input, prompt=self.prompt, uninv_input=self.uninv_input)
        else:
            self.output = self.netG(x=self.input, uninv_input=self.uninv_input)

        loss = 0
        zshape = self.output[:, 3:, :, :].shape
        ########################
        # Quantization before feeding into the invertible model
        LR = self.Quantization(self.output[:, :3, :, :])
        
        if compress_aware:
            print('integrating jpeg compression')
            LR_compressed = self.Compression(LR).to(self.device)
            
        ########################
        gaussian_scale = self.train_opt['gaussian_scale'] if self.train_opt['gaussian_scale'] != None else 1
        g_batch = self.gaussian_batch(zshape)
        y0 = torch.cat((LR, gaussian_scale * g_batch), dim=1)
        y1 = torch.cat((LR_compressed, gaussian_scale * g_batch), dim=1) if compress_aware else None

        self.fake_H = self.netG(x=y0, rev=True)
        self.fake_H_compressed = self.netG(x=y1, rev=True) if compress_aware else None

        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            l_forw= self.loss_forward(self.output, self.ref_L.detach(), self.output[:, 3:, :, :])
            # l_back_rec, l_back_fea, l_back_gan = self.loss_backward(self.real_H, self.fake_H)
            l_back= self.loss_backward(self.real_H, self.fake_H)
            if compress_aware:
                cw = 1. # jpeg weight
                rw = 1. - cw # png weight
                l_back_compressed= self.loss_backward(self.real_H, self.fake_H_compressed)
                l_back = {k: l_back.get(k, .0) * rw + l_back_compressed.get(k, .0) * cw for k in set(l_back)}
            

            # l_forw_fit + l_back_rec + l_forw_ce + l_forw_lpips + l_back_lpips
            loss += l_forw.get('l_forw_fit', 0.0) + l_back.get('l_back_rec', 0.0) + l_forw.get('l_forw_ce', 0.0) + l_forw.get('l_forw_lpips', 0.0) + l_back.get('l_back_lpips', 0.0)
            

            loss.backward()

            # gradient clipping
            if self.train_opt['gradient_clipping']:
                nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])

            self.optimizer_G.step()

        # D
        # for p in self.netD.parameters():
        #     p.requires_grad = True

        # self.optimizer_D.zero_grad()
        # l_d_total = 0
        # pred_d_real = self.netD(self.real_H)
        # pred_d_fake = self.netD(self.fake_H.detach())
        # if self.opt['train']['gan_type'] == 'gan':
        #     l_d_real = self.cri_gan(pred_d_real, True)
        #     l_d_fake = self.cri_gan(pred_d_fake, False)
        #     l_d_total = l_d_real + l_d_fake
        # elif self.opt['train']['gan_type'] == 'ragan':
        #     l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
        #     l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), False)
        #     l_d_total = (l_d_real + l_d_fake) / 2

        # l_d_total.backward()
        # self.optimizer_D.step()

        # set log
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            for k, v in l_forw.items():
                self.log_dict[k] = v.item()
            for k, v in l_back.items():
                self.log_dict[k] = v.item()
            # self.log_dict['l_back_fea'] = l_back_fea.item()
            # self.log_dict['l_back_gan'] = l_back_gan.item()
        # self.log_dict['l_d'] = l_d_total.item()

    def predict(self):
        # 输入为ref_L, predict rev=True的结果
        self.netG.eval()

        Lshape = self.ref_L.shape

        input_dim = Lshape[1]
        # print('input_dim:', input_dim)
        self.input = self.real_H

        zshape = [Lshape[0], input_dim * (self.opt['scale']*2) - Lshape[1], Lshape[2], Lshape[3]]

        # print('zshape:', zshape)

        gaussian_scale = 1
        if self.test_opt and self.test_opt['gaussian_scale'] != None:
            gaussian_scale = self.test_opt['gaussian_scale']

        with torch.no_grad():
            self.forw_L = self.ref_L
            y_forw = torch.cat((self.forw_L, gaussian_scale * self.gaussian_batch(zshape)), dim=1)
            self.fake_H = self.netG(x=y_forw, rev=True)[:, :3, :, :]
        self.netG.train()

    def test(self, compress_flag=False):
        Lshape = self.ref_L.shape

        input_dim = Lshape[1]
        # print('input_dim:', input_dim)
        self.input = self.real_H

        zshape = [Lshape[0], input_dim * (self.opt['scale']*2) - Lshape[1], Lshape[2], Lshape[3]]

        # print('zshape:', zshape)

        gaussian_scale = 1
        if self.test_opt and self.test_opt['gaussian_scale'] != None:
            gaussian_scale = self.test_opt['gaussian_scale']

        self.netG.eval()
        with torch.no_grad():
            if self.prompt is not None:
                self.forw_L = self.netG(x=self.input, prompt=self.prompt, uninv_input=self.uninv_input)[:, :3, :, :]
            else:
                self.forw_L = self.netG(x=self.input, uninv_input=self.uninv_input)[:, :3, :, :]
            self.forw_L = self.Quantization(self.forw_L)
            if compress_flag:
                # print('using jpg compression')
                # print('before saving forw_L min max:', self.forw_L.min().item(), self.forw_L.max().item())
                # print(self.forw_L)
                # save forw_L for using cv2.imwrite
                save_path_tmp = 'tmp_forw_L.jpg'
                tmp_forw_L_img = util.tensor2img(self.forw_L)
                # util.save_img(tmp_forw_L_img, save_path_tmp, quality=95)
                util.save_img(tmp_forw_L_img, save_path_tmp)
                # Reread the tmp_forw_L_img:
                # print("intermediate tmp_forw_L_img before saving:")
                # print(tmp_forw_L_img)
                tmp_forw_L_img = cv2.imread(save_path_tmp)
                # print('intermediate tmp_forw_L_img after saving and reading:')
                # print(tmp_forw_L_img)
                # BGR to RGB
                tmp_forw_L_img = cv2.cvtColor(tmp_forw_L_img, cv2.COLOR_BGR2RGB)
                tmp_forw_L_img = tmp_forw_L_img.astype('float32') / 255.0
                tmp_forw_L_img = torch.from_numpy(np.transpose(tmp_forw_L_img, (2, 0, 1))).float().unsqueeze(0).to(self.device)
                # print('after jpg compression forw_L min max:', tmp_forw_L_img.min().item(), tmp_forw_L_img.max().item())
                # print(tmp_forw_L_img)
                g_batch = self.gaussian_batch(zshape)
                y_forw = torch.cat((tmp_forw_L_img, gaussian_scale * g_batch), dim=1)
                
                y_diffjpeg_forw = self.Compression(self.forw_L)
                y_diffjpeg_forw = torch.cat((y_diffjpeg_forw, gaussian_scale * g_batch), dim=1)
            else:
                y_forw = torch.cat((self.forw_L, gaussian_scale * self.gaussian_batch(zshape)), dim=1)
            
            self.fake_H = self.netG(x=y_forw, rev=True)[:, :3, :, :]
            self.fake_H_compressed = self.netG(x=y_diffjpeg_forw, rev=True)[:, :3, :, :] if compress_flag else None

        self.netG.train()

    def downscale(self, HR_img):
        self.netG.eval()
        with torch.no_grad():
            LR_img = self.netG(x=HR_img)[:, :3, :, :]
            LR_img = self.Quantization(self.forw_L)
        self.netG.train()

        return LR_img

    def upscale(self, LR_img, scale, gaussian_scale=1):
        Lshape = LR_img.shape
        zshape = [Lshape[0], Lshape[1] * (scale**2 - 1), Lshape[2], Lshape[3]]
        y_ = torch.cat((LR_img, gaussian_scale * self.gaussian_batch(zshape)), dim=1)

        self.netG.eval()
        with torch.no_grad():
            HR_img = self.netG(x=y_, rev=True)[:, :3, :, :]
        self.netG.train()

        return HR_img

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['LR_ref'] = self.ref_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        out_dict['LR'] = self.forw_L.detach()[0].float().cpu()
        out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        out_dict['SR_compressed'] = self.fake_H_compressed.detach()[0].float().cpu() if self.fake_H_compressed is not None else None
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

        # load_path_D = self.opt['path']['pretrain_model_D']
        # if load_path_D is not None:
        #     logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
        #     self.load_network(load_path_D, self.netD, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
        # self.save_network(self.netD, 'D', iter_label)
