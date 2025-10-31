import json
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

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
        data_json_file = dataset_opt.get('data_json_file', '/home/hesong/disk1/DF_INV/code/ControlNet-v1-1-nightly/dataset/OmniEdit-Filtered-1.2M/prompts.json')
        resolution = dataset_opt.get('resolution',512) #512
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

        source = cv2.imread(source_fpth)
        target = cv2.imread(target_fpth)
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
        return {
            'GT':source,
            'LQ':target,
            'prompt': prompt,
            'GT_path': source_fpth,
            'LQ_path': target_fpth
        }
