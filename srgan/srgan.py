
import argparse
import os
import numpy as np
import math
import itertools
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch
import cv2

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

os.makedirs('images', exist_ok=True)
os.makedirs('saved_models', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--dataset_path', type=str, default="/home/yuankunhao/tianchi/dataset", help='path of the dataset')
parser.add_argument('--batch_size', type=int, default=4, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
parser.add_argument('--hr_height', type=int, default=128, help='size of high res. image height')
parser.add_argument('--hr_width', type=int, default=128, help='size of high res. image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=1000, help='interval between sampling of images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=-1, help='interval between model checkpoints')
parser.add_argument('--parallel', type=bool, default=True, help='use multiple GPUs')
opt = parser.parse_args()




# Calculate output of image discriminator (PatchGAN)
def count_parameter(model):
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            if param.dim() > 1:
                print(f'{name}:{"x".join(str(x) for x in list(param.size()))}={num_param}')
            else:
                print(f'{name}:{num_param}')
            total_param += num_param
    return total_param

def main():
    cuda = torch.cuda.is_available()
    # patch_h, patch_w = int(opt.hr_height / 2**4), int(opt.hr_width / 2**4)
    # patch = [opt.batch_size, 1, patch_h, patch_w]

    # Initialize generator and discriminator
    generator = GeneratorResNet()
    discriminator = Discriminator()
    feature_extractor = FeatureExtractor()

    feature_extractor.eval()

    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_content = torch.nn.L1Loss()


    
    

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        feature_extractor = feature_extractor.cuda()
        criterion_GAN = criterion_GAN.cuda()
        criterion_content = criterion_content.cuda()

        # Optimizers
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

        # wrap model to automated mixed precision
        generator, optimizer_G = amp.initialize(generator, optimizer_G, opt_level="O1")
        discriminator, optimizer_D = amp.initialize(discriminator, optimizer_D, opt_level="O1")
        
        if opt.parallel:
            generator = nn.DataParallel(generator, device_ids=range(torch.cuda.device_count()))
            discriminator = nn.DataParallel(discriminator, device_ids=range(torch.cuda.device_count()))
            feature_extractor = nn.DataParallel(feature_extractor, device_ids=range(torch.cuda.device_count()))
            # criterion_GAN = nn.DataParallel(criterion_GAN, device_ids=range(torch.cuda.device_count()))
            # criterion_content = nn.DataParallel(criterion_content, device_ids=range(torch.cuda.device_count()))
            #print(f'Gen:{count_parameter(generator)},Dis:{count_parameter(discriminator)},VGG:{count_parameter(feature_extractor)}')

    if opt.checkpoint != 0:
        # Load pretrained models
        generator.load_state_dict(torch.load('saved_models/generator_%d.pth'))
        discriminator.load_state_dict(torch.load('saved_models/discriminator_%d.pth'))
    # else:
    #     # Initialize weights
    #     generator.apply(weights_init_normal)
    #     discriminator.apply(weights_init_normal)


    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # input_lr = Tensor(opt.batch_size, opt.channels, opt.hr_height//4, opt.hr_width//4)
    # input_hr = Tensor(opt.batch_size, opt.channels, opt.hr_height, opt.hr_width)
    # Adversarial ground truths

    # Transforms for low resolution images and high resolution images
    lr_transforms = [   transforms.Resize((opt.hr_height//4, opt.hr_width//4), Image.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) 
                    ]

    hr_transforms = [   transforms.Resize((opt.hr_height, opt.hr_width), Image.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) 
                    ]

    dataloader = DataLoader(ImageDataset(opt.dataset_path, lr_transforms=lr_transforms, hr_transforms=hr_transforms),
                            batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

    # ----------
    #  Training
    # ----------
    global_G_loss = 100.
    global_D_loss = 100.
    for epoch in range(opt.checkpoint, opt.n_epochs):
        epoch_G_loss = 0. 
        epoch_D_loss = 0.    
        for i, imgs in enumerate(dataloader):
            # Configure model input
            imgs_lr = imgs['lr']
            imgs_hr = imgs['hr']
            if cuda:
                imgs_lr = imgs_lr.cuda()
                imgs_hr = imgs_hr.cuda()

            # ------------------
            #  Train Generators
            # ------------------

            optimizer_G.zero_grad()

            # Generate a high resolution image from low resolution input
            gen_hr = generator(imgs_lr)

            # Adversarial loss
            N = imgs_hr.shape[0] #incase of last batch size
            gen_validity = discriminator(gen_hr)
            valid = Tensor(np.ones(N))
            fake = Tensor(np.zeros(N))
            # print(gen_validity, valid)
            loss_GAN = criterion_GAN(gen_validity, valid)

            # Content loss
            gen_features = feature_extractor(gen_hr)
            real_features = feature_extractor(imgs_hr)
            loss_content =  criterion_content(gen_features, real_features.detach())

            # Total loss
            loss_G = torch.mean(loss_content + 1e-3 * loss_GAN)
            with amp.scale_loss(loss_G, optimizer_G) as scaled_loss:
                scaled_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss of real and fake images

            loss_real = criterion_GAN(discriminator(imgs_hr), valid)
            loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

            # Total loss
            loss_D = torch.mean((loss_real + loss_fake) / 2)

            with amp.scale_loss(loss_D, optimizer_D) as scaled_loss:
                scaled_loss.backward()
            optimizer_D.step()


            # --------------
            #  Log Progress
            # --------------
            if i == opt.sample_interval:
                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                                                                    (epoch, opt.n_epochs, i, len(dataloader),
                                                                    loss_D.item(), loss_G.item()))
            epoch_G_loss += loss_G.item()
            epoch_D_loss += loss_D.item()
            
            # batches_done = epoch * len(dataloader) + i
            # if batches_done % opt.sample_interval == 0:
            #     # Save image sample
            #     gen_hr = (gen_hr + 1) / 2 * 255
            #     imgs_hr= (imgs_hr + 1) / 2 * 255

            #     gen_hr = gen_hr[0].cpu().data.numpy()
            #     imgs_hr = imgs_hr[0].cpu().data.numpy()

            #     gen_hr = np.transpose(gen_hr, axes=(1,2,0))
            #     imgs_hr = np.transpose(imgs_hr, axes=(1,2,0))
            #     cv2.imwrite('images/%d_gen.png' % batches_done,gen_hr)
            #     cv2.imwrite('images/%d_ori.png' % batches_done,imgs_hr)
        epoch_D_loss /= len(dataloader)
        epoch_G_loss /= len(dataloader)
        if epoch_D_loss + epoch_G_loss <= global_D_loss + global_G_loss:
            # Save model checkpoints
            global_D_loss = epoch_D_loss
            global_G_loss = epoch_G_loss
            torch.save(generator.state_dict(), 'saved_models/generator_%d.pth' % epoch)
            torch.save(discriminator.state_dict(), 'saved_models/discriminator_%d.pth' % epoch)

if __name__ == "__main__":
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    main()
