import numpy as np
import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, num_channels, t, s):
        super().__init__()
        self.num_channels = num_channels
        self.s = s
        self.t = t
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, t*num_channels, kernel_size=1, stride=1, padding=0), 
            nn.BatchNorm2d(t*num_channels), 
            nn.ReLU()
        )
        self.dwconv = nn.Sequential(
            nn.Conv2d(t*num_channels, t*num_channels, kernel_size=3, stride=self.s, padding=1, groups=t*num_channels), 
            nn.Conv2d(t*num_channels, t*num_channels, kernel_size=1, stride=1, padding=0), 
            nn.BatchNorm2d(t*num_channels), 
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(t*num_channels, num_channels, kernel_size=1, stride=1, padding=0), 
            nn.BatchNorm2d(num_channels)
        )
        if self.s == 2:
            self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.act = nn.ReLU()
    
    def forward(self, x):
        z = self.conv1(x)
        z = self.dwconv(z)
        z = self.conv2(z)
        if self.s == 2:
            x = self.avgpool(x)
        x = x + z
        x = self.act(x)
        return x


class PPM(nn.Module):
    def __init__(self, height, width, in_channels, bins):
        super().__init__()
        self.bins = bins
        self.convs = nn.ModuleList()
        for b in bins:
            self.convs.append(
                nn.Sequential(
                    nn.AvgPool2d(kernel_size=(height//b, width//b), stride=(height//b, width//b), padding=0), 
                    nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1), 
                    nn.BatchNorm2d(1), 
                    nn.Upsample(size=(height, width), mode="nearest")
                )
            )
        self.act = nn.ReLU()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels+len(bins), in_channels, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(in_channels), 
            nn.ReLU()
        )
    
    def forward(self, x):
        out = [x]
        for conv in self.convs:
            out.append(conv(x))
        x = torch.cat(out, 1)
        x = self.act(x)
        x = self.conv(x)
        return x


class FastSCN(nn.Module):
    def __init__(self, img_height, img_width, num_classes):
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes
        self.learning_to_downsample = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), 
                nn.BatchNorm2d(32), 
                nn.ReLU()
            ), 
            nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, groups=32), 
                nn.Conv2d(32, 48, kernel_size=1, stride=1, padding=0), 
                nn.BatchNorm2d(48), 
                nn.ReLU()
            ), 
            nn.Sequential(
                nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1, groups=48), 
                nn.Conv2d(48, 64, kernel_size=1, stride=1, padding=0), 
                nn.BatchNorm2d(64), 
                nn.ReLU()
            )
        )
        self.global_feature_extractor = nn.Sequential(
            Bottleneck(64, 6, 2), 
            Bottleneck(64, 6, 1), 
            Bottleneck(64, 6, 1), 
            nn.Sequential(
                nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1), 
                nn.BatchNorm2d(96), 
                nn.ReLU()
            ), 
            Bottleneck(96, 6, 2), 
            Bottleneck(96, 6, 1), 
            Bottleneck(96, 6, 1), 
            nn.Sequential(
                nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1), 
                nn.BatchNorm2d(128), 
                nn.ReLU()
            ), 
            Bottleneck(128, 6, 1), 
            Bottleneck(128, 6, 1), 
            Bottleneck(128, 6, 1), 
            PPM(self.img_height//32, self.img_width//32, 128, [8, 4, 2])
        )
        self.feature_fusion_gfe_branch = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="nearest"), 
            nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128), 
                nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0), 
                nn.BatchNorm2d(128), 
                nn.ReLU()
            ), 
            nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), 
                nn.BatchNorm2d(128), 
                nn.ReLU()
            )
        )
        self.feature_fusion_lds_branch = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(128), 
            nn.ReLU()
        )
        self.clsf = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128), 
                nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0), 
                nn.BatchNorm2d(128), 
                nn.ReLU()
            ), 
            nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128), 
                nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0), 
                nn.BatchNorm2d(128), 
                nn.ReLU()
            ), 
            nn.Conv2d(128, self.num_classes, kernel_size=3, stride=1, padding=1), 
            nn.Upsample(size=(self.img_height, self.img_width), mode="bilinear", align_corners=True)
        )
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.learning_to_downsample(x)
        y = self.global_feature_extractor(x)
        y = self.feature_fusion_gfe_branch(y)
        x = self.feature_fusion_lds_branch(x)
        x = x + y
        x = self.clsf(x)
        if self.training:
            x = self.log_softmax(x)
        return x