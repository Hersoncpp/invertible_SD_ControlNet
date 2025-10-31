import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class InvBlockExp(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=1.):
        super(InvBlockExp, self).__init__()

        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            y1 = x1 + self.F(x2)
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        else:
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)

        return torch.cat((y1, y2), 1)

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)

        return jac / x.shape[0]

class InvBlockTriad(nn.Module):
    def __init__(self, subnet_constructor, cover_channel_num, secret_channel_num, resolution_channel_num, clamp=1.):
        super(InvBlockExp, self).__init__()

        self.cover_channel_num = cover_channel_num
        self.secret_channel_num = secret_channel_num
        self.resolution_channel_num = resolution_channel_num

        self.clamp = clamp
        
        # cs, cr, rs
        self.F = {"sc": subnet_constructor(secret_channel_num, cover_channel_num),
                  "cr": subnet_constructor(cover_channel_num, resolution_channel_num),
                  "rs": subnet_constructor(resolution_channel_num, secret_channel_num)}
        
        self.G = {"cs": subnet_constructor(cover_channel_num, secret_channel_num),
                  "rc": subnet_constructor(resolution_channel_num, cover_channel_num),
                  "sr": subnet_constructor(secret_channel_num, resolution_channel_num)}
        
        self.H = {"cs": subnet_constructor(cover_channel_num, secret_channel_num),
                  "rc": subnet_constructor(resolution_channel_num, cover_channel_num),
                  "sr": subnet_constructor(secret_channel_num, resolution_channel_num)}
        
    def branch_forward(self, x1, x2, net_type, net_id, rev=False):
        if net_type == 'F':
            net = self.F[net_id]
            if not rev:
                y2 = x2 + net(x1)
            else:
                y2 = x2 - net(x1) 
        elif net_type == 'G':
            net = self.G[net_id]
            if not rev:
                y2 = x2 + net(x1)
            else:
                y2 = x2 - net(x1)
        elif net_type == 'H':
            net = self.H[net_id]
            if not rev:
                s = self.clamp * (torch.sigmoid(net(x1)) * 2 - 1)
                y2 = x2.mul(torch.exp(s))
            else:
                s = self.clamp * (torch.sigmoid(net(x1)) * 2 - 1)
                y2 = x2.div(torch.exp(s))
                
        return x1, y2

    def forward(self, c, s, r, rev=False):

        if not rev:
            c, r = self.branch_forward(c, r, 'F', 'cr', rev)
            r, s = self.branch_forward(r, s, 'F', 'rs', rev)
            s, c = self.branch_forward(s, c, 'F', 'sc', rev)
            c, s = self.branch_forward(c, s, 'H', 'cs', rev)
            s, r = self.branch_forward(s, r, 'H', 'sr', rev)
            r, c = self.branch_forward(r, c, 'H', 'rc', rev)
            c, s = self.branch_forward(c, s, 'G', 'cs', rev)
            s, r = self.branch_forward(s, r, 'G', 'sr', rev)
            r, c = self.branch_forward(r, c, 'G', 'rc', rev)
        else:
            r, c = self.branch_forward(r, c, 'G', 'rc', rev)
            s, r = self.branch_forward(s, r, 'G', 'sr', rev)
            c, s = self.branch_forward(c, s, 'G', 'cs', rev)
            r, c = self.branch_forward(r, c, 'H', 'rc', rev)
            s, r = self.branch_forward(s, r, 'H', 'sr', rev)
            c, s = self.branch_forward(c, s, 'H', 'cs', rev)
            s, c = self.branch_forward(s, c, 'F', 'sc', rev)
            r, s = self.branch_forward(r, s, 'F', 'rs', rev)
            c, r = self.branch_forward(c, r, 'F', 'cr', rev)            

        return c, s, r

    def jacobian(self, x, rev=False):
        return 1

class InvRescaleNet(nn.Module):
    def __init__(self, channel_in=3, channel_out=3, subnet_constructor=None, block_num=[], down_num=2, non_inv_block = None, fusion_submodule_type = None):
        super(InvRescaleNet, self).__init__()

        operations = []
        operations_final = []
        current_channel = channel_in
        
        # without SR
        if down_num == 0:
            channel_out = 1
            for j in range(block_num[0]):
                b = InvBlockExp(subnet_constructor, current_channel, channel_out)
                operations.append(b)

            for j in range(block_num[1]):
                b = InvBlockExp(fusion_submodule_type, 6, 3)
                operations_final.append(b)

        self.operations = nn.ModuleList(operations)
        self.operations_final = nn.ModuleList(operations_final)
        # print("#### non_inv_block:", non_inv_block)
        if non_inv_block is None:
            bBranch = [nn.Conv2d(channel_in, channel_in*4, kernel_size=5, stride=1, padding=2, bias=True), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel_in*4, channel_in*8, kernel_size=3, stride=2, padding=1, bias=True), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel_in*8, channel_in*16, kernel_size=3, stride=2, padding=1, bias=True), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(channel_in*16, channel_in*8, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(channel_in*8, channel_in*4, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel_in*4, channel_in, kernel_size=5, stride=1, padding=2, bias=True)]
            self.uninvBranch = nn.Sequential(*bBranch)
        elif non_inv_block == 'Identity':
            self.uninvBranch = nn.Identity()
        else:
            from cldm.ddim_hacked import DDIMSampler
            from annotator.util import resize_image, HWC3
            import einops
            import cv2
            # self.uninvBranch = non_inv_block['model']
            self.uninvInput = non_inv_block['input']
            non_inv_block['model'] = non_inv_block['model'].to(torch.device("cuda"))
            ddim_sampler = DDIMSampler(non_inv_block['model'])

            def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
                # input_image, prompt, image_resolution=256, ddim_steps20
                with torch.no_grad():
                    # input image is a tensor of shape [B, C, H, W], B=1
                    input_image = input_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    prompt = prompt[0]
                    #转成np.uint8
                    if input_image.dtype != np.uint8:
                        input_image = (input_image * 255).round().astype(np.uint8)
                    # print('input_image shape:', input_image.shape)
                    # print('dtype:', input_image.dtype)
                    input_image = HWC3(input_image)
                    detected_map = input_image.copy()
                    img = resize_image(input_image, image_resolution)
                    H, W, C = img.shape
                    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
                    
                    control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
                    control = torch.cat([control for _ in range(num_samples)], dim=0)
                    if num_samples == 1:
                        control = control.unsqueeze(0)
                    # print('control shape:', control.shape)
                    # print(num_samples)
                    control = einops.rearrange(control, 'b h w c -> b c h w').clone()
                    
                    if seed == -1:
                        seed = 12345 #np.random.randint(0, 65535)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)

                    cond = {"c_concat": [control], "c_crossattn": [non_inv_block['model'].get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
                    un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [non_inv_block['model'].get_learned_conditioning([n_prompt] * num_samples)]}
                    shape = (4, H // 8, W // 8)

                    non_inv_block['model'].control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
                    # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

                    samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                                 shape, cond, verbose=False, eta=eta,
                                                                 unconditional_guidance_scale=scale,
                                                                 unconditional_conditioning=un_cond)

                    x_samples = non_inv_block['model'].decode_first_stage(samples)
                    # x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
                    # print('x_samples shape:', x_samples.shape)
                return x_samples

            self.uninvBranch = process

        

    def forward(self, x, rev=False, cal_jacobian=False, prompt=None, uninv_input=None):
        # x is tensor of shape [B, C, H, W]
        # if rev == True:
        #     print("in reverse, x shape:", x.shape)
        out = x
        jacobian = 0

        if not rev:
            for op in self.operations:
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
                    
            if uninv_input is not None:
                out_uninv = self.uninvBranch(uninv_input)
            elif type(self.uninvBranch) == nn.Sequential:
                out_uninv = self.uninvBranch(x)
            else:
                input_dict = self.uninvInput.copy()
                input_dict['input_image'] = x
                input_dict['prompt'] = prompt
                out_uninv = self.uninvBranch(**input_dict)

            # print('out shape:', out.shape, 'out_uninv shape:', out_uninv.shape)
            out_ =  torch.cat((out, out_uninv ), 1)   
            
            for op in self.operations_final:
                out_ = op.forward(out_, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out_, rev)      
            return out_   
             
        else:
            for op in reversed(self.operations_final):
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
        
            out_ = out[:,:3,:,:]
            for op in reversed(self.operations):
                out_ = op.forward(out_, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out_, rev)

        if cal_jacobian:
            return out_, jacobian
        else:
            return out_

class InvRescaleNetD(nn.Module):
    def __init__(self, channel_in=3, channel_out=3, subnet_constructor=None, block_num=[], down_num=2, non_inv_block = None, fusion_submodule_type = None):
        super(InvRescaleNetD, self).__init__()

        operations_cover = []
        operations_secret = []
        operations_final = []
        current_channel = channel_in
        
        # without SR
        if down_num == 0:
            channel_out = 1
            for j in range(block_num[0]):
                b = InvBlockExp(subnet_constructor, current_channel, channel_out)
                operations_secret.append(b)
                
            for j in range(block_num[1]):
                b = InvBlockExp(subnet_constructor, current_channel, channel_out)
                operations_cover.append(b)

            for j in range(block_num[2]):
                b = InvBlockExp(fusion_submodule_type, 6, 3)
                operations_final.append(b)

        self.operations_cover = nn.ModuleList(operations_cover)
        self.operations_secret = nn.ModuleList(operations_secret)
        self.operations_final = nn.ModuleList(operations_final)
        # print("#### non_inv_block:", non_inv_block)
        if non_inv_block is None:
            bBranch = [nn.Conv2d(channel_in, channel_in*4, kernel_size=5, stride=1, padding=2, bias=True), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel_in*4, channel_in*8, kernel_size=3, stride=2, padding=1, bias=True), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel_in*8, channel_in*16, kernel_size=3, stride=2, padding=1, bias=True), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(channel_in*16, channel_in*8, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(channel_in*8, channel_in*4, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel_in*4, channel_in, kernel_size=5, stride=1, padding=2, bias=True)]
            self.uninvBranch = nn.Sequential(*bBranch)
        elif non_inv_block == 'Identity':
            self.uninvBranch = nn.Identity()
        else:
            from cldm.ddim_hacked import DDIMSampler
            from annotator.util import resize_image, HWC3
            import einops
            import cv2
            # self.uninvBranch = non_inv_block['model']
            self.uninvInput = non_inv_block['input']
            non_inv_block['model'] = non_inv_block['model'].to(torch.device("cuda"))
            ddim_sampler = DDIMSampler(non_inv_block['model'])

            def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
                # input_image, prompt, image_resolution=256, ddim_steps20
                with torch.no_grad():
                    # input image is a tensor of shape [B, C, H, W], B=1
                    input_image = input_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    prompt = prompt[0]
                    #转成np.uint8
                    if input_image.dtype != np.uint8:
                        input_image = (input_image * 255).round().astype(np.uint8)
                    # print('input_image shape:', input_image.shape)
                    # print('dtype:', input_image.dtype)
                    input_image = HWC3(input_image)
                    detected_map = input_image.copy()
                    img = resize_image(input_image, image_resolution)
                    H, W, C = img.shape
                    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
                    
                    control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
                    control = torch.cat([control for _ in range(num_samples)], dim=0)
                    if num_samples == 1:
                        control = control.unsqueeze(0)
                    # print('control shape:', control.shape)
                    # print(num_samples)
                    control = einops.rearrange(control, 'b h w c -> b c h w').clone()
                    
                    if seed == -1:
                        seed = 12345 #np.random.randint(0, 65535)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)

                    cond = {"c_concat": [control], "c_crossattn": [non_inv_block['model'].get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
                    un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [non_inv_block['model'].get_learned_conditioning([n_prompt] * num_samples)]}
                    shape = (4, H // 8, W // 8)

                    non_inv_block['model'].control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
                    # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

                    samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                                 shape, cond, verbose=False, eta=eta,
                                                                 unconditional_guidance_scale=scale,
                                                                 unconditional_conditioning=un_cond)

                    x_samples = non_inv_block['model'].decode_first_stage(samples)
                    # x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
                    # print('x_samples shape:', x_samples.shape)
                return x_samples

            self.uninvBranch = process

        

    def forward(self, x, rev=False, cal_jacobian=False, prompt=None, uninv_input=None):
        # x is tensor of shape [B, C, H, W]
        # if rev == True:
        #     print("in reverse, x shape:", x.shape)
        out = x
        jacobian = 0

        if not rev: 
            for op in self.operations_secret:
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
                    
            if uninv_input is not None:
                out_uninv = self.uninvBranch(uninv_input)
            elif type(self.uninvBranch) == nn.Sequential:
                out_uninv = self.uninvBranch(x)
            else:
                input_dict = self.uninvInput.copy()
                input_dict['input_image'] = x
                input_dict['prompt'] = prompt
                out_uninv = self.uninvBranch(**input_dict)

            for op in self.operations_cover:
                out_uninv = op.forward(out_uninv, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out_uninv, rev)
            
            # print('out shape:', out.shape, 'out_uninv shape:', out_uninv.shape)
            out_ =  torch.cat((out, out_uninv ), 1)   
            
            for op in self.operations_final:
                out_ = op.forward(out_, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out_, rev)      
            return out_   
             
        else:
            for op in reversed(self.operations_final):
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
        
            out_secret = out[:,:3,:,:]
            out_cover = out[:,3:,:,:]
            for op in reversed(self.operations_secret):
                out_secret = op.forward(out_secret, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out_secret, rev)
            out_ = out_secret
            
            for op in reversed(self.operations_cover):
                out_cover = op.forward(out_cover, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out_cover, rev)

        if cal_jacobian:
            return out_, jacobian
        else:
            return out_

class InvNetJPEGAware(nn.Module):
    def __init__(self, channel_in=3, channel_out=3, subnet_constructor=None, block_num=8, down_num=2, non_inv_block = None):
        super(InvNetJPEGAware, self).__init__()
        operations = []
      
        # without SR
        for _ in range(block_num[0]):
            b = InvBlockTriad(subnet_constructor, 3, 3, 3)
            operations.append(b)

        self.operations = nn.ModuleList(operations)
        print("#### non_inv_block:", non_inv_block)
        if non_inv_block is None:
            bBranch = [nn.Conv2d(channel_in, channel_in*4, kernel_size=5, stride=1, padding=2, bias=True), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel_in*4, channel_in*8, kernel_size=3, stride=2, padding=1, bias=True), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel_in*8, channel_in*16, kernel_size=3, stride=2, padding=1, bias=True), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(channel_in*16, channel_in*8, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(channel_in*8, channel_in*4, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel_in*4, channel_in, kernel_size=5, stride=1, padding=2, bias=True)]
            self.uninvBranch = nn.Sequential(*bBranch)
        elif non_inv_block == 'Identity':
            self.uninvBranch = nn.Identity()
        else:
            from cldm.ddim_hacked import DDIMSampler
            from annotator.util import resize_image, HWC3
            import einops
            import cv2
            # self.uninvBranch = non_inv_block['model']
            self.uninvInput = non_inv_block['input']
            non_inv_block['model'] = non_inv_block['model'].to(torch.device("cuda"))
            ddim_sampler = DDIMSampler(non_inv_block['model'])

            def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
                # input_image, prompt, image_resolution=256, ddim_steps20
                with torch.no_grad():
                    # input image is a tensor of shape [B, C, H, W], B=1
                    input_image = input_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    prompt = prompt[0]
                    #转成np.uint8
                    if input_image.dtype != np.uint8:
                        input_image = (input_image * 255).round().astype(np.uint8)
                    # print('input_image shape:', input_image.shape)
                    # print('dtype:', input_image.dtype)
                    input_image = HWC3(input_image)
                    detected_map = input_image.copy()
                    img = resize_image(input_image, image_resolution)
                    H, W, C = img.shape
                    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
                    
                    control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
                    control = torch.cat([control for _ in range(num_samples)], dim=0)
                    if num_samples == 1:
                        control = control.unsqueeze(0)
                    # print('control shape:', control.shape)
                    # print(num_samples)
                    control = einops.rearrange(control, 'b h w c -> b c h w').clone()
                    
                    if seed == -1:
                        seed = 12345 #np.random.randint(0, 65535)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)

                    cond = {"c_concat": [control], "c_crossattn": [non_inv_block['model'].get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
                    un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [non_inv_block['model'].get_learned_conditioning([n_prompt] * num_samples)]}
                    shape = (4, H // 8, W // 8)

                    non_inv_block['model'].control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
                    # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

                    samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                                 shape, cond, verbose=False, eta=eta,
                                                                 unconditional_guidance_scale=scale,
                                                                 unconditional_conditioning=un_cond)

                    x_samples = non_inv_block['model'].decode_first_stage(samples)
                    # x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
                    # print('x_samples shape:', x_samples.shape)
                return x_samples

            self.uninvBranch = process

        

    def forward(self, x, r, rev=False, cal_jacobian=False, prompt=None, uninv_input=None):
        # x is tensor of shape [B, C, H, W]
        # if rev == True:
        #     print("in reverse, x shape:", x.shape)
        out = x

        if not rev:
                    
            if uninv_input is not None:
                y = self.uninvBranch(uninv_input)
            elif type(self.uninvBranch) == nn.Sequential:
                y = self.uninvBranch(x)
            else:
                input_dict = self.uninvInput.copy()
                input_dict['input_image'] = x
                input_dict['prompt'] = prompt
                y = self.uninvBranch(**input_dict)

            # print('out shape:', out.shape, 'out_uninv shape:', out_uninv.shape)
            
            for op in self.operations:
                x, y, r = op.forward(x, y, r, rev)  
            return x, y, r  
             
        else:
            for op in reversed(self.operations_final):
                out = op.forward(out, rev)
        
            out_ = out[:,:3,:,:]
            for op in reversed(self.operations):
                out_ = op.forward(out_, rev)
        
        return out_