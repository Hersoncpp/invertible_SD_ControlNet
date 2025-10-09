import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import config as c
from natsort import natsorted
import cv2
from torchvision import transforms
import json

def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class My_Dataset(Dataset):
    def __init__(self, data_json_file='./dataset/OmniEdit-Filtered-1.2M/prompts.json', resolution=c.cropsize):
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
        # source = source.float() / 255.0

        # Normalize target images to [-1, 1].
        # target = target.float() / 255.0

        # reshape from [H, W, C] to [C, H, W]
        #print("source, target shape:")
        #print(source.shape, target.shape)
        # source = source.permute(1, 2, 0)
        # target = target.permute(1, 2, 0)

        return {
            'source': source,
            'target': target,
            'prompt': prompt
        }

class Hinet_Dataset(Dataset):
    def __init__(self, transforms_=None, mode="train"):

        self.transform = transforms_
        self.mode = mode
        if mode == 'train':
            # train
            self.files = natsorted(sorted(glob.glob(c.TRAIN_PATH + "/*." + c.format_train)))
        else:
            # test
            self.files = sorted(glob.glob(c.VAL_PATH + "/*." + c.format_val))

    def __getitem__(self, index):
        try:
            image = Image.open(self.files[index])
            image = to_rgb(image)
            item = self.transform(image)
            return item

        except:
            return self.__getitem__(index + 1)

    def __len__(self):
        if self.mode == 'shuffle':
            return max(len(self.files_cover), len(self.files_secret))

        else:
            return len(self.files)


# transform = T.Compose([
#     T.RandomHorizontalFlip(),
#     T.RandomVerticalFlip(),
#     T.RandomCrop(c.cropsize),
#     T.ToTensor()
# ])

# transform_val = T.Compose([
#     T.CenterCrop(c.cropsize_val),
#     T.ToTensor(),
# ])


# # Training data loader
# trainloader = DataLoader(
#     Hinet_Dataset(transforms_=transform, mode="train"),
#     batch_size=c.batch_size,
#     shuffle=True,
#     pin_memory=True,
#     num_workers=8,
#     drop_last=True
# )
# # Test data loader
# testloader = DataLoader(
#     Hinet_Dataset(transforms_=transform_val, mode="val"),
#     batch_size=c.batchsize_val,
#     shuffle=False,
#     pin_memory=True,
#     num_workers=1,
#     drop_last=True
# )
        
trainloader = DataLoader(My_Dataset(data_json_file=c.TRAIN_JSON_PATH, resolution=c.cropsize), batch_size=c.batch_size, shuffle=True, num_workers=8, drop_last=True)

testloader = DataLoader(My_Dataset(data_json_file=c.VAL_JSON_PATH, resolution=c.cropsize_val), batch_size=c.batchsize_val, shuffle=False, num_workers=1, drop_last=True)