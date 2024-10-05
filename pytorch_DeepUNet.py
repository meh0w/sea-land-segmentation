import torch
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn as nn

class bn_relu(nn.Module):
    """ 
    """
    def __init__(self, channels):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class conv_bn_relu(nn.Module):
    """ 
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, (3,3), (1,1), 'valid')
        self.bn_relu = bn_relu(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn_relu(x)
        return x
    
class down_block(nn.Module):
    """ 
    """
    def __init__(self, in_channels):
        super().__init__()
        self.bn_relu1 = bn_relu(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, in_channels*2, (3,3), (1,1), 'same')
        self.bn_relu2 = bn_relu(in_channels*2)
        self.conv2 = torch.nn.Conv2d(in_channels*2, in_channels, (3,3), (1,1), 'same')
        self.maxpool = torch.nn.MaxPool2d((2, 2), (2, 2))

    def forward(self, inputs):
        x = self.bn_relu1(inputs)
        x = self.conv1(x)
        x = self.bn_relu2(x)
        x = self.conv2(x)

        plus_out = x + inputs

        out = self.maxpool(plus_out)

        return plus_out, out
    
class up_block(nn.Module):
    """ 
    """
    def __init__(self, in_channels):
        super().__init__()
        self.upsample = torch.nn.Upsample(scale_factor=2)
        self.bn_relu1 = bn_relu(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels*2, in_channels*2, (3,3), (1,1), 'same')
        self.bn_relu2 = bn_relu(in_channels*2)
        self.conv2 = torch.nn.Conv2d(in_channels*2, in_channels, (3,3), (1,1), 'same')

    def forward(self, inputs, plus_in):
        up_sampled = self.upsample(inputs)
        x = self.bn_relu1(up_sampled)
        x = torch.cat([x, plus_in], dim=1)
        x = self.conv1(x)
        x = self.bn_relu2(x)
        x = self.conv2(x)

        out = x + up_sampled

        return out
    

class DeepUNet(nn.Module):
    """ 
    """
    def __init__(self, in_channels, scale, outputs=1):
        super().__init__()
        self.outputs = outputs
        self.conv1 = torch.nn.Conv2d(in_channels, 32//scale, (3,3), (1,1), 'same')

        self.down1 = down_block(32//scale)
        self.down2 = down_block(32//scale)
        self.down3 = down_block(32//scale)
        self.down4 = down_block(32//scale)
        self.down5 = down_block(32//scale)
        self.down6 = down_block(32//scale)
        self.down7 = down_block(32//scale)

        self.up7 = up_block(32//scale)
        self.up6 = up_block(32//scale)
        self.up5 = up_block(32//scale)
        self.up4 = up_block(32//scale)
        self.up3 = up_block(32//scale)
        self.up2 = up_block(32//scale)
        self.up1 = up_block(32//scale)

        self.maxpool = torch.nn.MaxPool2d((2, 2), (2, 2))
        self.bn_relu = bn_relu(32//scale)
        self.conv2 = torch.nn.Conv2d(32//scale, 2, (1,1), (1,1), 'valid')
        if self.outputs == 2:
            self.conv3 = torch.nn.Conv2d(32//scale, 2, (1,1), (1,1), 'valid')

    def forward(self, inputs):
        x = self.conv1(inputs)
        down_plus1, down1 = self.down1(x)
        down_plus2, down2 = self.down2(down1)
        down_plus3, down3 = self.down3(down2)
        down_plus4, down4 = self.down4(down3)
        down_plus5, down5 = self.down5(down4)
        down_plus6, down6 = self.down6(down5)
        down_plus7, down7 = self.down7(down6)

        up7 = self.up7(down7, down_plus7)
        up6 = self.up6(up7, down_plus6)
        up5 = self.up5(up6, down_plus5)
        up4 = self.up4(up5, down_plus4)
        up3 = self.up3(up4, down_plus3)
        up2 = self.up2(up3, down_plus2)
        up1 = self.up1(up2, down_plus1)

        # x = self.maxpool(up1)
        x = self.bn_relu(up1)
        seg = self.conv2(x)
        if self.outputs == 2:
            edge = self.conv3(x)
            return seg, edge
        else:
            return seg