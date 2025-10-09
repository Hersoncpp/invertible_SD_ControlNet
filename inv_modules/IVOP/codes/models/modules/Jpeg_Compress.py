import torch
import torch.nn as nn
import cv2
import numpy as np


def extract_from_tensor_batch(tensor_batch):
    # tensor size is (B, C, H, W)
    B, C, H, W = tensor_batch.shape
    tensors = []
    for i in range(B):
        tensors.append(tensor_batch[i, :, :, :])
    return tensors

class Jpeg_Compress(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input):
        steg_img = input
        # print('shape of steg_img before jpeg compress:', steg_img.shape)
        tmp_step_img_path = "/home/hesong/disk1/DF_INV/code/ControlNet-v1-1-nightly/inv_modules/IVOP/codes/models/modules/tmp.jpg"
        # 存储为jpg格式
        def transform_steg_img(steg_img):
            return torch.clamp(steg_img, 0, 1)

        tensors = extract_from_tensor_batch(transform_steg_img(steg_img))
        compressed_tensors = []
        for i in range(len(tensors)):
            img_np = tensors[i].squeeze().cpu().numpy().transpose(1, 2, 0) * 255.
            img_np = img_np.astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(tmp_step_img_path, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            img_bgr = cv2.imread(tmp_step_img_path)
            steg_img_jpg = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            steg_img_jpg = steg_img_jpg.transpose(2, 0, 1) / 255.0
            steg_img_jpg = torch.from_numpy(steg_img_jpg).unsqueeze(0).float().cuda()
            compressed_tensors.append(steg_img_jpg)
        compressed_steg_img = torch.cat(compressed_tensors, dim=0)
        # print('shape of steg_img after jpeg compress:', compressed_steg_img.shape)
        return compressed_steg_img
    @staticmethod
    def backward(ctx, grad_output):
        # gradient pass through as identity function
        return grad_output
    
class Jpeg_Compress_Layer(nn.Module):
    def __init__(self):
        super(Jpeg_Compress_Layer, self).__init__()

    def forward(self, input):
        return Jpeg_Compress.apply(input)