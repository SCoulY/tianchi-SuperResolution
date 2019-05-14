
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

os.makedirs('images', exist_ok=True)
os.makedirs('saved_models', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--dataset_path', type=str, default="./dataset", help='path of the dataset')
parser.add_argument('--batch_size', type=int, default=16, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
parser.add_argument('--hr_height', type=int, default=128, help='size of high res. image height')
parser.add_argument('--hr_width', type=int, default=128, help='size of high res. image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=10, help='interval between sampling of images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=-1, help='interval between model checkpoints')
opt = parser.parse_args()




# Calculate output of image discriminator (PatchGAN)


def main():
    cuda = True if torch.cuda.is_available() else False
    patch_h, patch_w = int(opt.hr_height / 2**4), int(opt.hr_width / 2**4)
    patch = [opt.batch_size, 1, patch_h, patch_w]

    # Initialize generator and discriminator
    generator = GeneratorResNet()
    discriminator = Discriminator()
    feature_extractor = FeatureExtractor()

    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_content = torch.nn.L1Loss()

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        feature_extractor = feature_extractor.cuda()
        criterion_GAN = criterion_GAN.cuda()
        criterion_content = criterion_content.cuda()

    if opt.epoch != 0:
        # Load pretrained models
        generator.load_state_dict(torch.load('saved_models/generator_%d.pth'))
        discriminator.load_state_dict(torch.load('saved_models/discriminator_%d.pth'))
    else:
        # Initialize weights
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    # input_lr = Tensor(opt.batch_size, opt.channels, opt.hr_height//4, opt.hr_width//4)
    # input_hr = Tensor(opt.batch_size, opt.channels, opt.hr_height, opt.hr_width)
    # Adversarial ground truths

    # Transforms for low resolution images and high resolution images
    lr_transforms = [   transforms.Resize((opt.hr_height//4, opt.hr_height//4), Image.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) 
                    ]

    hr_transforms = [   transforms.Resize((opt.hr_height, opt.hr_height), Image.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) 
                    ]

    dataloader = DataLoader(ImageDataset(opt.dataset_path, lr_transforms=lr_transforms, hr_transforms=hr_transforms),
                            batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

    # ----------
    #  Training
    # ----------

    for epoch in range(opt.epoch, opt.n_epochs):
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
            patch[0] = imgs_hr.shape[0] #incase of last batch size
            valid = Tensor(np.ones(patch))
            fake = Tensor(np.zeros(patch))
            gen_validity = discriminator(gen_hr)
            loss_GAN = criterion_GAN(gen_validity, valid)

            # Content loss
            gen_features = feature_extractor(gen_hr)
            real_features = feature_extractor(imgs_hr)
            loss_content =  criterion_content(gen_features, real_features.detach())

            # Total loss
            loss_G = loss_content + 1e-3 * loss_GAN

            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss of real and fake images

            loss_real = criterion_GAN(discriminator(imgs_hr), valid)
            loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

            # Total loss
            loss_D = (loss_real + loss_fake) / 2

            loss_D.backward()
            optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                                                                (epoch, opt.n_epochs, i, len(dataloader),
                                                                loss_D.item(), loss_G.item()))

            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                # Save image sample
                gen_hr = (gen_hr + 1) / 2 * 255
                imgs_hr= (imgs_hr + 1) / 2 * 255

                gen_hr = gen_hr[0].cpu().data.numpy()
                imgs_hr = imgs_hr[0].cpu().data.numpy()

                gen_hr = np.transpose(gen_hr, axes=(1,2,0))
                imgs_hr = np.transpose(imgs_hr, axes=(1,2,0))
                cv2.imwrite('images/%d_gen.png' % batches_done,gen_hr)
                cv2.imwrite('images/%d_ori.png' % batches_done,imgs_hr)



        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), 'saved_models/generator_%d.pth' % epoch)
            torch.save(discriminator.state_dict(), 'saved_models/discriminator_%d.pth' % epoch)

if __name__ == "__main__":
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
