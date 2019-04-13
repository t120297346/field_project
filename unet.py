# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 20:33:44 2019

@author: user
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class double_conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, padding = 1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace = True),
                nn.Conv2d(out_channel, out_channel, 3, padding = 1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace = True),
                )
            

class inconv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(inconv, self).__init__()
        self.conv = double_conv(in_channel, out_channel)
    
    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_channel, out_channel)
                )
        def forward(self, x):
            x = self.mpconv(x)
            return x

class up(nn.Module):
    def __init__(self, in_channel, out_channel, bilinear = True):
        super(up, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)
        else:
            self.up = nn.ConvTranspose2d(in_channel // 2, out_channel // 2, 2, stride = 2)
        
        self.conv = double_conv(in_channel, out_channel)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, (diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2))
        
        x = torch.cat([x2, x1], dim = 1)
        x = self.conv
        return x
    
class outconv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(outconv, self).__init__() 
        self.conv = nn.Conv2d(in_channel, out_channel, 1)
        
    def forward(self, x):
        x = self.conv(x)
        return x
        
        
            
            
            
            
            
            
            
            
            
            