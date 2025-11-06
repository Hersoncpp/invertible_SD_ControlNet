import json
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset


def get_text_embedding_model(model_name):
    from cldm.model import create_model, load_state_dict
    model = create_model(f'./models/{model_name}.yaml').cpu()
    model_name = "control_v11e_sd15_ip2p-finetuned_1"
    model.load_state_dict(load_state_dict(f'./models/{model_name}.pth', location=f'cuda:'), strict=False)
    model = model.to(torch.device('cuda'))
    return model.get_learned_conditioning

class STDataset(Dataset):
    """
    Style Transfer Dataset
    Args:
        dataset_opt: dataset options, including data_json_file, resolution, etc.
    Returns:
        data: dataset, including GT, LQ, prompt, GT_path, LQ_path
    """
    def __init__(self, dataset_opt):
        super(STDataset, self).__init__()
        self.opt = dataset_opt
        self.data = []
        data_json_file = dataset_opt.get('data_json_file', '/home/hwangem/invSD/invertible_SD_ControlNet/dataset/OmniEdit-Filtered-1.2M_train/prompts.json')
        resolution = dataset_opt.get('resolution',512) #512
        self.text_embedding_model = dataset_opt.get('text_embedding_model', None)
        if self.text_embedding_model is not None:
            # text_embedding_model is a model name
            self.get_text_embedding_model = get_text_embedding_model(self.text_embedding_model)
        else:
            self.get_text_embedding_model = None

        with open(data_json_file, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

        self.resolution = resolution
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor()
        ])
        self.conditioning_image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor()
        ])
        print(f'Loaded {len(self.data)} data from {data_json_file}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_fpth = item['source']
        target_fpth = item['target']
        prompt = item['prompt']
        if self.get_text_embedding_model is not None:
            text_embedding = self.get_text_embedding_model([prompt])[0]
        else:
            text_embedding = None
        # print('source_fpth:', source_fpth)
        # print('target_fpth:', target_fpth)
        # print('prompt:', prompt)

        source = cv2.imread(source_fpth)
        target = cv2.imread(target_fpth)

        # check if source and target are empty
        if source is None or target is None:
            print(f'source or target is empty at index {idx}, with path {source_fpth} and {target_fpth}')
            return self.__getitem__(idx + 1)

        # print("source, target fpth:", source_fpth, target_fpth)
        # print("#######################################source--before############################################")
        # print(source.shape)
        # print(source)
        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB).copy()
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB).copy()


        source = self.conditioning_image_transform(source)
        target = self.image_transform(target)
        # print("#######################################source--mid############################################")
        # print(source.shape)
        # print(source)

        # Normalize source images to [0, 1]. and source is of type torch.FloatTensor
        # source = source.float() / 255.0

        # Normalize target images to [0, 1].
        # target = target.float() / 255.0
        # print("#######################################source--after############################################")
        # print(source.shape)
        # print(source)
        
        # reshape from [H, W, C] to [C, H, W]
        #print("source, target shape:")
        #print(source.shape, target.shape)
        #source = source.permute(1, 2, 0)
        #target = target.permute(1, 2, 0)
        #print('source, target shape after permute:')
        #print(source.shape, target.shape)
        if text_embedding is not None:
            return {
                'GT':source,
                'LQ':target,
                'prompt': prompt,
                'GT_path': source_fpth,
                'LQ_path': target_fpth,
                'text_embedding': text_embedding
            }
        else:
            return {
                'GT':source,
                'LQ':target,
                'prompt': prompt,
                'GT_path': source_fpth,
                'LQ_path': target_fpth,
            }
