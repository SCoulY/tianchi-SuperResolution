import torch
import torch.nn as nn
import torch.nn.functional as F

cfg = [[512, 512, 512, 512], [512, 512, 512, 256], [256, 256, 256, 128] ,[128, 128, 64] ,[64, 64, 'F']]

class segnet(nn.Module):
    def __init__(self, encoder, init_weights=True, pretrain=True):
        super(segnet, self).__init__()
        
        # get the layers from a pre-trained vgg16 model
        self.en_bloc1 = encoder[0:6]
        self.en_bloc2 = encoder[7:13]
        self.en_bloc3 = encoder[14:23]
        self.en_bloc4 = encoder[24:33]
        self.en_bloc5 = encoder[34:43]
        
        self.de_bloc5 = self.make_decoder(cfg[0], batch_norm=True)
        self.de_bloc4 = self.make_decoder(cfg[1], batch_norm=True)
        self.de_bloc3 = self.make_decoder(cfg[2], batch_norm=True)
        self.de_bloc2 = self.make_decoder(cfg[3], batch_norm=True)
        self.de_bloc1 = self.make_decoder(cfg[4], batch_norm=True)

        if init_weights:
            self._initialize_weights(pretrain)

    def forward(self, x):
        
        # encoder with fixed vgg parameters
        x = self.en_bloc1(x) 
        x, id1 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)   
        x = self.en_bloc2(x)
        x, id2 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        x = self.en_bloc3(x)
        x, id3 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        x = self.en_bloc4(x)
        x, id4 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        x = self.en_bloc5(x)
        x, id5 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        
        #decoder with learnable parameters
        x = F.max_unpool2d(x, id5, kernel_size=2, stride=2, output_size=id4.size())
        x = self.de_bloc5(x)
        x = F.max_unpool2d(x, id4, kernel_size=2, stride=2, output_size=id3.size())
        x = self.de_bloc4(x)
        x = F.max_unpool2d(x, id3, kernel_size=2, stride=2, output_size=id2.size())
        x = self.de_bloc3(x)
        x = F.max_unpool2d(x, id2, kernel_size=2, stride=2, output_size=id1.size())
        x = self.de_bloc2(x)
        x = F.max_unpool2d(x, id1, kernel_size=2, stride=2)
        x = self.de_bloc1(x)
        
        return x

    def make_decoder(self, cfg, num_classes=5, batch_norm=False):
        layers = []
        in_channels = cfg[0]
        for v in cfg[1:]:
            #if v == 'U':
            #    layers += [nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)]                
            #else:
            if v == 'F':
                trans_conv2d = nn.ConvTranspose2d(in_channels, num_classes, kernel_size=3, padding=1)
                layers += [trans_conv2d]
            else:
                trans_conv2d = nn.ConvTranspose2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [trans_conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [trans_conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    
    def _initialize_weights(self, pretrain):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if not pretrain:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):         
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)