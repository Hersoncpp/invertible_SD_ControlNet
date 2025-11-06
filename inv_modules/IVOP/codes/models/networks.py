import torch
import logging
import models.modules.discriminator_vgg_arch as SRGAN_arch
from models.modules.Inv_arch import *
from models.modules.Subnet_constructor import subnet
from models.modules.decompresser import Decompresser
import sys
sys.path.append('/home/hesong/disk1/DF_INV/code/ControlNet-v1-1-nightly')
from cldm.model import create_model, load_state_dict
import math
logger = logging.getLogger('base')


####################
# define network
####################
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    subnet_type = which_model['subnet_type']
    fusion_submodule_type = which_model.get('fusion_submodule_type', subnet_type)
    print("subnet_type:", subnet_type)
    print("fusion_submodule_type:", fusion_submodule_type)
    non_inv_block_model_name = which_model.get('non_inv_block', None)
    if opt_net['init']:
        init = opt_net['init']
    else:
        init = 'xavier'

    down_num = int(math.log(opt_net['scale'], 2))

    def read_model_from_pth(model_name):
        learning_rate = 1e-5
        sd_locked = True
        only_mid_control = False

        model_dir = '/home/hesong/disk1/DF_INV/code/ControlNet-v1-1-nightly/models'
        model_config = model_name.split('-')[0]
        model = create_model(f'{model_dir}/{model_config}.yaml').cpu()
        model.load_state_dict(load_state_dict(f'{model_dir}/v1-5-pruned.ckpt', location='cuda'), strict=False)
        model.load_state_dict(load_state_dict(f'{model_dir}/{model_name}.pth', location='cuda'), strict=False)

        model.learning_rate = learning_rate
        model.sd_locked = sd_locked
        model.only_mid_control = only_mid_control
        
        # freeze the entire model
        for param in model.parameters():
            param.requires_grad = False

        return model

    if non_inv_block_model_name is not None:
        if non_inv_block_model_name != 'Identity':
            non_inv_block = {} 
            non_inv_block['model'] = read_model_from_pth(non_inv_block_model_name)
            # input_image, prompt, image_resolution=256, ddim_steps20
            # a_prompt = 'best quality'
            # n_prompt = 'lowres, bad anatomy, bad hands, cropped, worst quality'
            non_inv_block['input'] = {
                'input_image': None,
                'prompt': None,
                'a_prompt': 'best quality',
                'n_prompt': 'lowres, bad anatomy, bad hands, cropped, worst quality',
                'num_samples': 1,
                'image_resolution': 256,
                'ddim_steps': 10,
                'guess_mode': False,
                'strength': 1.0,
                'scale': 9.0,
                'seed': 12345,
                'eta': 0.0
            }
        else:
            non_inv_block = 'Identity'

    else:
        non_inv_block = None
    print(opt_net['branch_type'])
    if opt_net['branch_type'] == 'dual_branch':
        print("### Using InvRescaleNetD ###")
        netG = InvRescaleNetD(opt_net['in_nc'], opt_net['out_nc'], subnet(subnet_type, init), opt_net['block_num'], down_num, non_inv_block=non_inv_block, fusion_submodule_type=subnet(fusion_submodule_type, init))
    else:
        print("### Using InvRescaleNet ###")
        netG = InvRescaleNet(opt_net['in_nc'], opt_net['out_nc'], subnet(subnet_type, init), opt_net['block_num'], down_num, non_inv_block=non_inv_block, fusion_submodule_type=subnet(fusion_submodule_type, init))
    return netG



#### Discriminator
def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = SRGAN_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD


#### Define Network used for Perceptual Loss
def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = SRGAN_arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF

### Artifact Remover Network
def define_AR(opt):
    netAR = Decompresser(channel=3, block_type='CBAM', init='xavier')
    return netAR