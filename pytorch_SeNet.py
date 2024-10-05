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
        self.conv = torch.nn.Conv2d(in_channels, out_channels, (3,3), (1,1), 'same')
        self.bn_relu = bn_relu(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn_relu(x)
        return x
    
class deconv_bn_relu(nn.Module):
    """ 
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = torch.nn.ConvTranspose2d(in_channels, out_channels, (3,3), (1,1), 1)
        self.bn_relu = bn_relu(out_channels)

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn_relu(x)
        return x
    
class SeNet(nn.Module):
    """ 
    """
    def __init__(self, in_channels, scale, outputs=2):
        super().__init__()
        self.outputs = outputs
        self.cbnr_1 = conv_bn_relu(in_channels, 32//scale)
        self.cbnr_2 = conv_bn_relu(32//scale, 64//scale)
        self.cbnr_3 = conv_bn_relu(64//scale, 64//scale)
        self.cbnr_4 = conv_bn_relu(64//scale, 128//scale)
        self.cbnr_5 = conv_bn_relu(128//scale, 128//scale)
        self.cbnr_6 = conv_bn_relu(128//scale, 128//scale)
        self.cbnr_7 = conv_bn_relu(128//scale, 128//scale)

        self.dcbnr_1 = deconv_bn_relu(128//scale, 128//scale)
        self.dcbnr_2 = deconv_bn_relu(128//scale, 128//scale)
        self.dcbnr_3 = deconv_bn_relu(128//scale, 128//scale)
        self.dcbnr_4 = deconv_bn_relu(128//scale, 64//scale)
        self.dcbnr_5 = deconv_bn_relu(64//scale, 64//scale)
        self.dcbnr_6 = deconv_bn_relu(64//scale, 32//scale)
        self.dcbnr_7 = deconv_bn_relu(32//scale, 32//scale)

        self.maxpool1 = torch.nn.MaxPool2d((2, 2), (2, 2), return_indices=True)
        self.maxpool2 = torch.nn.MaxPool2d((2, 2), (2, 2), return_indices=True)
        self.maxpool3 = torch.nn.MaxPool2d((2, 2), (2, 2), return_indices=True)

        self.unpool1 = torch.nn.MaxUnpool2d((2, 2), (2, 2))
        self.unpool2 = torch.nn.MaxUnpool2d((2, 2), (2, 2))
        self.unpool3 = torch.nn.MaxUnpool2d((2, 2), (2, 2))

        self.conv2 = torch.nn.Conv2d(32//scale, 2, (1,1), (1,1), 'valid')
        if self.outputs == 2:
            self.conv3 = torch.nn.Conv2d(32//scale, 2, (1,1), (1,1), 'valid')

    def forward(self, inputs):
        cbnr_1 = self.cbnr_1(inputs)
        cbnr_2 = self.cbnr_2(cbnr_1)

        size1 = cbnr_2.size()
        maxp_1, maxp_idx_1 = self.maxpool1(cbnr_2)
        
        cbnr_3 = self.cbnr_3(maxp_1)
        cbnr_4 = self.cbnr_4(cbnr_3)

        size2 = cbnr_4.size()
        maxp_2, maxp_idx_2 = self.maxpool2(cbnr_4)
        
        cbnr_5 = self.cbnr_5(maxp_2)
        cbnr_6 = self.cbnr_6(cbnr_5)
        cbnr_7 = self.cbnr_7(cbnr_6)

        size3 = cbnr_7.size()
        maxp_3, maxp_idx_3 = self.maxpool3(cbnr_7)
        
        umaxp_1 = self.unpool1(maxp_3, maxp_idx_3, output_size=size3)

        dcbnr_1 = self.dcbnr_1(umaxp_1)
        dcbnr_2 = self.dcbnr_2(dcbnr_1)
        dcbnr_3 = self.dcbnr_3(dcbnr_2)

        umaxp_2 = self.unpool2(dcbnr_3, maxp_idx_2, output_size=size2)

        dcbnr_4 = self.dcbnr_4(umaxp_2)
        dcbnr_5 = self.dcbnr_5(dcbnr_4)

        umaxp_3 = self.unpool3(dcbnr_5, maxp_idx_1, output_size=size1)

        dcbnr_6 = self.dcbnr_6(umaxp_3)
        dcbnr_7 = self.dcbnr_7(dcbnr_6)

        seg = self.conv2(dcbnr_7)
        if self.outputs == 2:
            edge = self.conv3(dcbnr_7)
            return seg, edge
        else:
            return seg
