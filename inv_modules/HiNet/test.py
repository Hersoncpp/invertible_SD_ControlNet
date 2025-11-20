import math
import torch
import torch.nn
import torch.optim
import torchvision
import numpy as np
from model import *
import config as c
import datasets
import modules.Unet_common as common
from tqdm import tqdm
import cv2

class Quant(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        input = torch.clamp(input, 0, 1)
        output = (input * 255.).round() / 255.
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class Quantization(nn.Module):
    def __init__(self):
        super(Quantization, self).__init__()

    def forward(self, input):
        return Quant.apply(input)

class jpg_Compress(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        global device
        steg_img = input

        tmp_step_img_path = "/home/hesong/disk1/DF_INV/code/ControlNet-v1-1-nightly/inv_modules/HiNet/image/tmp.jpg"
        # 存储为jpg格式
        def transform_steg_img(steg_img):
            return torch.clamp(steg_img, 0, 1)

        img_np = transform_steg_img(steg_img).squeeze().cpu().numpy().transpose(1, 2, 0) * 255.
        img_np = img_np.astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(tmp_step_img_path, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        img_bgr = cv2.imread(tmp_step_img_path)
        steg_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        steg_img = steg_img.transpose(2, 0, 1) / 255.0
        steg_img = torch.from_numpy(steg_img).unsqueeze(0).float().to(device)
        return steg_img
    @staticmethod
    def backward(ctx, grad_output):
        # gradient pass through as identity function
        return grad_output
#set default device to be our first cuda device
torch.cuda.set_device(c.device_ids[0])
device = torch.device(f"cuda:{c.device_ids[0]}" if torch.cuda.is_available() else "cpu")


def load(name):
    state_dicts = torch.load(name)
    network_state_dict = {k:v for k,v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')


def gauss_noise(shape):

    noise = torch.zeros(shape).to(device)
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).to(device)

    return noise


def computePSNR(origin,pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin/1.0 - pred/1.0) ** 2 )
    if mse < 1.0e-10:
      return 100
    return 10 * math.log10(255.0**2/mse)


net = Model()
net.to(device)
init_model(net)
net = torch.nn.DataParallel(net, device_ids=c.device_ids)
params_trainable = (list(filter(lambda p: p.requires_grad, net.parameters())))
optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step, gamma=c.gamma)

load(c.MODEL_PATH + c.suffix)

net.eval()

dwt = common.DWT()
iwt = common.IWT()
quantization_fun = Quantization()
compress_flag = True
with torch.no_grad():
    for i, data in tqdm(enumerate(datasets.testloader)):
        # data = data.to(device)
        # cover = data[data.shape[0] // 2:, :, :, :]
        # secret = data[:data.shape[0] // 2, :, :, :]
        cover = data['target'].to(device)
        secret = data['source'].to(device)
        # print(cover)
        # print("cover min max:")
        # print(cover.max())
        cover_input = dwt(cover)
        secret_input = dwt(secret)
        input_img = torch.cat((cover_input, secret_input), 1)

        #################
        #    forward:   #
        #################
        output = net(input_img)
        output_steg = output.narrow(1, 0, 4 * c.channels_in)
        output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
        steg_img = iwt(output_steg)
##########################
        quantize_flag = True
        # print("max min of steg_img:", steg_img.max(), steg_img.min())
        # print(steg_img)
        # quantization
        if quantize_flag:
            steg_img = quantization_fun(steg_img)
            # print("After quantization:")
            # print("max min of steg_img:", steg_img.max(), steg_img.min())
            # print(steg_img)
        if compress_flag:
            tmp_step_img_path = "/home/hesong/disk1/DF_INV/code/ControlNet-v1-1-nightly/inv_modules/HiNet/image/tmp.jpg"
            # tmp_tor_step_img_path = "/home/hesong/disk1/DF_INV/code/ControlNet-v1-1-nightly/inv_modules/HiNet/image/tmp_tor.jpg"
            # tmp_tor_trans_img_path = "/home/hesong/disk1/DF_INV/code/ControlNet-v1-1-nightly/inv_modules/HiNet/image/tmp_trans.jpg"
            # torchvision.utils.save_image(steg_img, tmp_tor_step_img_path)

            # 输出steg_img的形状:
            # print("steg_img shape:", steg_img.shape)
            # 存储为jpg格式
            def transform_steg_img(steg_img):
                # steg_min, steg_max = steg_img.min(), steg_img.max()
                # # normalize to 0-1 by min-max
                # return (steg_img - steg_min) / (steg_max - steg_min)
                return torch.clamp(steg_img, 0, 1)
            # clamped_steg_img = torch.clamp(steg_img, 0, 1)
            # torchvision.utils.save_image(clamped_steg_img, tmp_tor_trans_img_path)
            # img_steg_tor = cv2.imread(tmp_tor_step_img_path)
            # img_steg_trans = cv2.imread(tmp_tor_trans_img_path)
            # diff_img = abs(img_steg_trans - img_steg_tor)
            # diff_img_path = "/home/hesong/disk1/DF_INV/code/ControlNet-v1-1-nightly/inv_modules/HiNet/image/diff.jpg"
            # cv2.imwrite(diff_img_path, diff_img)
            # print("max diff:", diff_img.max())
            img_np = transform_steg_img(steg_img).squeeze().cpu().numpy().transpose(1, 2, 0) * 255.
            img_np = img_np.astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(tmp_step_img_path, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            # cv2.imwrite(tmp_step_img_path, img_bgr)
            # # 重新读取jpg格式
            # if quantize_flag is not True:
                # img_bgr = cv2.imread(tmp_tor_step_img_path)
            img_bgr = cv2.imread(tmp_step_img_path)
            steg_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            steg_img = steg_img.transpose(2, 0, 1) / 255.0
            steg_img = torch.from_numpy(steg_img).unsqueeze(0).float().to(device)
            # print("max min of steg_img:", steg_img.max(), steg_img.min())
            # print(steg_img)
        output_steg = dwt(steg_img)
##########################
        backward_z = gauss_noise(output_z.shape)

        #################
        #   backward:   #
        #################
        output_rev = torch.cat((output_steg, backward_z), 1)
        bacward_img = net(output_rev, True)
        secret_rev = bacward_img.narrow(1, 4 * c.channels_in, bacward_img.shape[1] - 4 * c.channels_in)
        secret_rev = iwt(secret_rev)
        cover_rev = bacward_img.narrow(1, 0, 4 * c.channels_in)
        cover_rev = iwt(cover_rev)
        resi_cover = (steg_img - cover) * 20
        resi_secret = (secret_rev - secret) * 20
        img_type = 'png'   
        torchvision.utils.save_image(cover, c.IMAGE_PATH_cover + f'{c.save_suffix}_{i:05d}.' + img_type)
        torchvision.utils.save_image(secret, c.IMAGE_PATH_secret + f'{c.save_suffix}_{i:05d}.' + img_type)
        torchvision.utils.save_image(steg_img, c.IMAGE_PATH_steg + f'{c.save_suffix}_{i:05d}.' + img_type)
        torchvision.utils.save_image(secret_rev, c.IMAGE_PATH_secret_rev + f'{c.save_suffix}_{i:05d}.' + img_type)
        # break




