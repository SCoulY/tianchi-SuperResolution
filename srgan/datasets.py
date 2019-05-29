import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, lr_transforms=None, hr_transforms=None):
        self.lr_transform = transforms.Compose(lr_transforms)
        self.hr_transform = transforms.Compose(hr_transforms)
        self.train_folders = ['youku149', 'youku49', 'youku99']
        self.lr_files = sorted([x for x in glob.glob(os.path.join(root,'lr')+'/*/*.bmp', recursive=True) if 'youku199' not in x])  #exclude eval set
        self.gt_files = sorted([x for x in glob.glob(os.path.join(root,'gt')+'/*/*.bmp', recursive=True) if 'youku199' not in x])
        
    def __getitem__(self, index):
        img_lr = Image.open(self.lr_files[index])
        img_lr = self.lr_transform(img_lr)
        img_hr = Image.open(self.gt_files[index])
        img_hr = self.hr_transform(img_hr)
        return {'lr': img_lr, 'hr': img_hr}

    def __len__(self):
        return len(self.files)

if __name__ == "__main__":
    lr_transforms = [   transforms.Resize((270, 480), Image.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) 
                    ]

    hr_transforms = [   transforms.Resize((1080, 1920), Image.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) 
                    ]
    dataset = ImageDataset('F:/BaiduNetdiskDownload/优酷数据帧图像', lr_transforms=lr_transforms, hr_transforms=hr_transforms)
    for data in dataset:
        print(data['lr'].shape, data['hr'].shape)
        break