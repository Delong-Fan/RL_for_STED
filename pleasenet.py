import os
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        # 深度卷积：逐通道卷积
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride=stride,
                                   padding=((kernel_size - 1) // 2) * dilation,
                                   dilation=dilation,
                                   groups=in_channels,
                                   bias=False)
        # 逐点卷积：1×1卷积实现通道融合
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        out = self.relu(out)
        return out

class NewNetOptimized(nn.Module):
    def __init__(self, num_classes=4):
        super(NewNetOptimized, self).__init__()
        self.conv1 = DepthwiseSeparableConv(5, 48, kernel_size=7, stride=1,dilation=3)
        self.bn1 = nn.BatchNorm2d(48)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU(inplace=True)

        # 使用深度可分离卷积块
        self.conv2_ds = DepthwiseSeparableConv(48, 96, kernel_size=3, stride=2,dilation=3)   # 下采样到 H/4
        self.conv3_ds = DepthwiseSeparableConv(96, 128, kernel_size=3, stride=2,dilation=3)   # 下采样到 H/8
        self.conv4_ds = DepthwiseSeparableConv(128, 256, kernel_size=3, stride=1, dilation=2)  # 尺寸保持 H/8
        self.conv5_ds = DepthwiseSeparableConv(256, 256, kernel_size=3, stride=1, dilation=2) # 尺寸保持 H/8

        # ========== 2) FPN 横向分支 (Lateral Convs) ==========
        self.lat2 = nn.Conv2d(96,  128, kernel_size=1)   # 对应 c2
        self.lat3 = nn.Conv2d(128,  128, kernel_size=1)   # 对应 c3
        self.lat4 = nn.Conv2d(256, 128, kernel_size=1)   # 对应 c4
        self.lat5 = nn.Conv2d(256, 128, kernel_size=1)   # 对应 c5

        # ========== 3) FPN 输出平滑卷积 ==========
        self.out2 = DepthwiseSeparableConv(128, 128, kernel_size=3, stride=1)
        self.out3 = DepthwiseSeparableConv(128, 128, kernel_size=3, stride=1)
        self.out4 = DepthwiseSeparableConv(128, 128, kernel_size=3, stride=1)
        self.out5 = DepthwiseSeparableConv(128, 128, kernel_size=3, stride=1)

        # ========== 4) 多尺度融合 ==========
        self.fusion_conv = DepthwiseSeparableConv(512, 256, kernel_size=3, stride=1)
        self.output = nn.Conv2d(256,64,kernel_size=1,stride=1)

        # ========== 5) 最终分类 ==========
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                    nn.Linear(64, 4),
        )

    def forward(self, x):
        # Backbone 前向传播
        x = self.relu1(self.bn1(self.conv1(x)))  # (N,24,H/2,W/2)
        c2 = self.conv2_ds(x)                    # (N,48,H/4,W/4)
        c3 = self.conv3_ds(c2)                   # (N,96,H/8,W/8)
        c4 = self.conv4_ds(c3)                   # (N,128,H/8,W/8)
        c5 = self.conv5_ds(c4)                   # (N,128,H/8,W/8)

        # FPN 横向分支
        p5 = self.lat5(c5)   # (N,64,H/8,W/8)
        p4 = self.lat4(c4)   # (N,64,H/8,W/8)
        p3 = self.lat3(c3)   # (N,64,H/8,W/8)
        p2 = self.lat2(c2)   # (N,64,H/4,W/4)

        # 自顶向下融合
        p4 = p4 + p5                  # (N,64,H/8,W/8)
        p3 = p3 + p4                  # (N,64,H/8,W/8)
        p3_up = F.interpolate(p3, scale_factor=2, mode='nearest')  # (N,64,H/4,W/4)
        p2 = p2 + p3_up              # (N,64,H/4,W/4)

        # 对每个尺度进行平滑处理
        p5 = self.out5(p5)   # (N,64,H/8,W/8)
        p4 = self.out4(p4)   # (N,64,H/8,W/8)
        p3 = self.out3(p3)   # (N,64,H/8,W/8)
        p2 = self.out2(p2)   # (N,64,H/4,W/4)

        # 多尺度融合：将 p3, p4, p5 上采样到 p2 的尺寸，再进行拼接
        p3_up = F.interpolate(p3, scale_factor=2, mode='nearest')  # (N,64,H/4,W/4)
        p4_up = F.interpolate(p4, scale_factor=2, mode='nearest')  # (N,64,H/4,W/4)
        p5_up = F.interpolate(p5, scale_factor=2, mode='nearest')  # (N,64,H/4,W/4)
        fusion = torch.cat([p2, p3_up, p4_up, p5_up], dim=1)       # (N,256,H/4,W/4)
        fusion = self.fusion_conv(fusion)                          # (N,64,H/4,W/4)
        fusion = self.bn2(fusion)
        fusion = self.relu1(fusion)
        out = self.output(fusion)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        # 最终分类
        '''
        out = self.avgpool(fusion)   # (N,64,1,1)
        out = torch.flatten(out, 1)  # (N,64)
        out = self.fc(out)           # (N,num_classes)
        '''
        return out

def new(num_classes=4):
    return NewNetOptimized(num_classes)
