import json
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data_json_file='./dataset/OmniEdit-Filtered-1.2M/prompts.json', resolution=512):
        self.data = []
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

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB).copy()
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB).copy()

        source = self.conditioning_image_transform(source)
        target = self.image_transform(target)

        # Normalize source images to [0, 1]. and source is of type torch.FloatTensor
        source = source.float() / 255.0

        # Normalize target images to [-1, 1].
        target = (target.float() / 127.5) - 1.0

        # reshape from [H, W, C] to [C, H, W]
        #print("source, target shape:")
        #print(source.shape, target.shape)
        source = source.permute(1, 2, 0)
        target = target.permute(1, 2, 0)

        return dict(jpg=target, txt=prompt, hint=source)
