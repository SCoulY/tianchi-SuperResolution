from models import *
from datasets import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import cv2


cuda = torch.cuda.is_available()
generator = GeneratorResNet()

lr_transforms = [   
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) 
                ]


dataloader = DataLoader(PredDataset('/home/yuankunhao/tianchi/dataset/lr/youku199', lr_transforms=lr_transforms),
                        batch_size=1, shuffle=False, num_workers=4)


def unnormalize(x):
    return ((x+1)/2)*255

if cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    generator = generator.cuda().eval()
    train_dict = torch.load('saved_models/generator_4.pth') 
    predict_dict = {}
    for key in train_dict:
        predict_key = key[7:]
        predict_dict[predict_key] = train_dict[key]
    generator.load_state_dict(predict_dict)

write_path = '/home/yuankunhao/tianchi//dataset/youku199GT_gen'
os.makedirs(write_path, exist_ok=True)
for i, imgs in tqdm(enumerate(dataloader)):
    imgs_lr = imgs['lr']
    id = imgs['id'][0][:-4] + '_gen.bmp'
    if cuda:
        imgs_lr = imgs_lr.cuda()
    gen_hr = generator(imgs_lr).squeeze()
    gen_hr = unnormalize(gen_hr).data.cpu().numpy().astype(np.uint8)
    gen_hr = gen_hr.transpose(1,2,0)
    gen_hr = cv2.cvtColor(gen_hr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(write_path, id), gen_hr)