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
    def forward(ctx, input, tmp_file_name, quality=95):
        ctx.save_for_backward(input)
        ctx.tmp_file_name = tmp_file_name
        ctx.quality = quality

        steg_img = input
        tmp_step_img_path = f"/home/yukai/disk1/invertible_SD_ControlNet/inv_modules/IVOP/experiments/{tmp_file_name}.jpg"

        def transform_steg_img(steg_img):
            return torch.clamp(steg_img, 0, 1)

        tensors = extract_from_tensor_batch(transform_steg_img(steg_img))
        compressed_tensors = []
        for i in range(len(tensors)):
            img_np = tensors[i].squeeze().cpu().numpy().transpose(1, 2, 0) * 255.
            img_np = img_np.astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(tmp_step_img_path, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

            img_bgr = cv2.imread(tmp_step_img_path)
            steg_img_jpg = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            steg_img_jpg = steg_img_jpg.transpose(2, 0, 1) / 255.0
            steg_img_jpg = torch.from_numpy(steg_img_jpg).unsqueeze(0).float().cuda()
            compressed_tensors.append(steg_img_jpg)
        compressed_steg_img = torch.cat(compressed_tensors, dim=0)
        return compressed_steg_img

    @staticmethod
    def backward(ctx, grad_output):
        
        grad_input = grad_output
        
        grad_tmp_file_name = None
        grad_quality = None
        return grad_input, grad_tmp_file_name, grad_quality
    
class Jpeg_Compress_Layer(nn.Module):
    def __init__(self, tmp_file_name):
        super(Jpeg_Compress_Layer, self).__init__()
        self.tmp_file_name = tmp_file_name

    def forward(self, input):
        return Jpeg_Compress.apply(input, self.tmp_file_name)