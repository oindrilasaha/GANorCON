import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet_parts import *


class UpBlock(nn.Module):
    def __init__(self, inplanes, planes, upsample=False):
        super(UpBlock, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.will_ups = upsample

    def forward(self, x):
        if self.will_ups:
            x = nn.functional.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes=34, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, int(1024))
        self.down1 = Down(int(1024), int(256))
        self.down2 = Down(int(256), int(256))
        self.down3 = Down(int(256), int(512/2))
        self.down4 = Down(int(512/2), int(512))
        factor = 2 if bilinear else 1
        self.down5 = Down(int(512), int(1024) // factor)
        self.up0 = Up(int(1024), int(512) // factor, bilinear)
        self.up1 = Up(int(512), int(512) // factor, bilinear)
        self.up2 = Up(int(512), int(256) // factor, bilinear)
        self.up3 = Up(int(384), int(512) // factor, bilinear)
        self.up4 = Up(int(1280), int(256), bilinear)
        self.outc = OutConv(int(256), n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up0(x6, x5)
        
        x = self.up1(x, x4)
        
        x = self.up2(x, x3)
        x = self.up3(x, x2)

        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def init_weights(self, init_type='kaiming', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)



class fcn(nn.Module):
    def __init__(self, descriptor_dimension, n_classes=34):
        super().__init__()  
        self.decoder = nn.Sequential(nn.Conv2d(
            in_channels=descriptor_dimension, 
            out_channels=1024,
            kernel_size=1,
            padding=0,
            bias=False),
            nn.BatchNorm2d(1024), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, 
            out_channels=256,
            kernel_size=1,
            padding=0,
            bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, 
            out_channels=n_classes,
            kernel_size=1,
            padding=0,
            bias=False))


    def forward(self, input):

        out = self.decoder(input)
        return out

    def init_weights(self, init_type='kaiming', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

