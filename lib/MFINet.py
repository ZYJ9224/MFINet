import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import argparse
import numpy as np
from torch import nn
from torch.nn import init
from os.path import join
from lib.pvtv2 import pvt_v2_b2

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class MEU(nn.Module):
    def __init__(self, in_channel):
        super(MEU, self).__init__()
        out_channel = in_channel // 4
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.ca = CALayer(out_channel, 16)

        self.conv_cat = BasicConv2d(in_channel, 64, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, 64, 1)

    def forward(self, x):
        x_s = torch.chunk(x, 4, 1)

        x0 = self.branch0(x_s[0])
        x1 = self.branch1(x_s[1])
        x2 = self.branch2(x_s[2])
        x3 = self.branch3(x_s[3])

        x0 = torch.chunk(x0, 4, 1)
        x1 = torch.chunk(x1, 4, 1)
        x2 = torch.chunk(x2, 4, 1)
        x3 = torch.chunk(x3, 4, 1)

        x0_1 = self.ca(torch.cat([x0[0], x1[0], x2[0], x3[0]], 1))
        x1_1 = self.ca(torch.cat([x0[1], x1[1], x2[1], x3[1]], 1))
        x2_1 = self.ca(torch.cat([x0[2], x1[2], x2[2], x3[2]], 1))
        x3_1 = self.ca(torch.cat([x0[3], x1[3], x2[3], x3[3]], 1))

        x_cat = self.conv_cat(torch.cat((x0_1, x1_1, x2_1, x3_1), 1))
        out = x_cat + self.conv_res(x)
        return out


class GIEU(nn.Module):
    def __init__(self, channels, r=4):
        super(GIEU, self).__init__()
        out_channels = int(channels // r)

        # global_att
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )
        self.conv_1 = BasicConv2d(2*channels, channels, kernel_size=3, padding=1)
        self.sig = nn.Sigmoid()

    def forward(self, x3, x4):
        if x3.size()[2:] != x4.size()[2:]:
            x3 = F.interpolate(x3, size=x4.size()[2:], mode='bilinear')
        y_cat = self.conv_1(torch.cat([x3, x4], 1))

        y_att = self.global_att(y_cat)
        out = self.sig(y_att) * y_cat + x4

        return out


class SIEU(nn.Module):
    def __init__(self, channels, r=4):
        super(SIEU, self).__init__()
        out_channels = int(channels // r)

        # local_att
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )
        self.conv_1 = BasicConv2d(2*channels, channels, kernel_size=3, padding=1)
        self.sig = nn.Sigmoid()

    def forward(self, x2, x3, fg):
        if x3.size()[2:] != x2.size()[2:]:
            x3 = F.interpolate(x3, size=x2.size()[2:], mode='bilinear')
        if fg.size()[2:] != x2.size()[2:]:
            fg = F.interpolate(fg, size=x2.size()[2:], mode='bilinear')
            
        y_cat = self.conv_1(torch.cat([x2,x3], 1)) * fg
        
        y_att = self.local_att(y_cat)
        out = self.sig(y_att) * y_cat + x2
        return out


class SA1(nn.Module):
    def __init__(self, in_dim):
        super(SA1, self).__init__()
        self.conv1 = BasicConv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = BasicConv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        self.conv3 = BasicConv2d(2 * in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        # -----------最后交互----------------------#
        self.conv6 = BasicConv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        self.ca = CALayer(in_dim)

    def forward(self, x, y):
        # 初始特征融合
        z = x * y
        z = self.conv1(z)
        x = self.conv2(z + x)
        y = self.conv2(z + y)
        f_cat = torch.cat([x, y], 1)
        f = self.conv3(f_cat)
        f_att = self.ca(f)
        out_final = self.conv6(f_att)
        return out_final + f


class MFFM(nn.Module):
    def __init__(self, in_dim):
        super(MFFM, self).__init__()

        self.sa1 = SA1(in_dim)
        self.sa2 = SA1(in_dim)

        self.conv1 = BasicConv2d(2 * in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, xf, xg, xe):
        if xg.size()[2:] != xf.size()[2:]:
            xg = F.interpolate(xg, size=xf.size()[2:], mode='bilinear')
        if xe.size()[2:] != xf.size()[2:]:
            xe = F.interpolate(xe, size=xf.size()[2:], mode='bilinear')

        sa_fg = self.sa1(xf.mul(self.sigmoid(xg + xe)), xf)
        sa_fe = self.sa2(xf.mul(1 - self.sigmoid(xg + xe)), xf)
        final = self.conv1(torch.cat([sa_fg, sa_fe], 1)) + sa_fg
        return final


class CA(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        out = avg_out
        return self.sigmoid(out)


class SA(nn.Module):
    def __init__(self, kernel_size=7):
        super(SA, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = self.conv1(max_out)
        return self.sigmoid(x)



class CRM(nn.Module):
    def __init__(self, channel):
        super(CRM, self).__init__()
        self.query_conv = nn.Conv2d(channel, channel, kernel_size=1)
        self.key_conv = nn.Conv2d(channel, channel, kernel_size=1)
        self.value_conv_2 = nn.Conv2d(channel, channel, kernel_size=1)
        self.value_conv_3 = nn.Conv2d(channel, channel, kernel_size=1)
        self.gamma_2 = nn.Parameter(torch.zeros(1))
        self.gamma_3 = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        self.conv_2 = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv_3 = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv_out = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)
    def forward(self, x3, x2): 
        x3 = F.interpolate(x3, size=x2.size()[2:], mode='bilinear', align_corners=True)
        m_batchsize, C, height, width = x3.size()
        proj_query = self.query_conv(x3).view(m_batchsize, -1, width * height)
        proj_key = self.key_conv(x2).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key) #b,c,c

        attention = self.softmax(energy)

        proj_value_2 = self.value_conv_2(x2).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_value_3 = self.value_conv_3(x3).view(m_batchsize, -1, width * height).permute(0, 2, 1)

        
        out_2 = torch.bmm(proj_value_2, attention.permute(0, 2, 1))
        out_2 = out_2.view(m_batchsize, C, height, width)
        out_2 = self.conv_2(self.gamma_2 * out_2 + x2)

        out_3 = torch.bmm(proj_value_3, attention.permute(0, 2, 1))
        out_3 = out_3.view(m_batchsize, C, height, width)
        out_3 = self.conv_3(self.gamma_3 * out_3 + x3)

        x_out = self.conv_out(out_2 + out_3)
        return x_out


class Decoder(nn.Module):
    def __init__(self, channel):
        super(Decoder, self).__init__()
        self.cfu1 = CRM(channel)
        self.cfu2 = CRM(channel)
        self.cfu3 = CRM(channel)
        self.conv64_1 = nn.Conv2d(64,1,1)
    def forward(self, x1, x2, x3, x4):

        x34 = self.cfu1(x4, x3)
        x23 = self.cfu2(x34, x2)
        x12 = self.cfu3(x23, x1)

        S_34 = F.interpolate(self.conv64_1(x34), scale_factor=16,
                                    mode='bilinear')
        S_23 = F.interpolate(self.conv64_1(x23), scale_factor=8,
                                    mode='bilinear')
        S_12 = F.interpolate(self.conv64_1(x12), scale_factor=4,
                                    mode='bilinear')
        return S_34, S_23, S_12

class Network(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=64, imagenet_pretrained=True):
        super(Network, self).__init__()
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './lib/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.reduce4 = BasicConv2d(512, 64, kernel_size=1)
        self.reduce3 = BasicConv2d(320, 64, kernel_size=1)
        self.reduce2 = BasicConv2d(128, 64, kernel_size=1)
        self.reduce1 = BasicConv2d(64, 64, kernel_size=1)

        self.aspp4 = MEU(64)
        self.aspp3 = MEU(64)
        self.aspp2 = MEU(64)
        self.aspp1 = MEU(64)

        self.fg = GIEU(64)
        self.fs = SIEU(64)

        self.mffm1 = MFFM(64)
        self.mffm2 = MFFM(64)
        self.mffm3 = MFFM(64)
        self.mffm4 = MFFM(64)

        self.decoder = Decoder(64)
        self.downsample2 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.downsample4 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)
        self.downsample8 = nn.Upsample(scale_factor=0.125, mode='bilinear', align_corners=True)

    def forward(self, x):
        pvt = self.backbone(x)
        r1 = self.reduce1(pvt[0])
        r2 = self.reduce2(pvt[1])
        r3 = self.reduce3(pvt[2])
        r4 = self.reduce4(pvt[3])

        f4 = self.aspp4(r4)
        f3 = self.aspp3(r3)
        f2 = self.aspp2(r2)
        f1 = self.aspp1(r1)
        # 全局信息
        fg = self.fg(r3, r4)
        # 空间信息
        fs = self.fs(r2, r3, fg)

        f4 = self.mffm4(f4, fg, fs)
        f3 = self.mffm3(f3, fg, fs)
        f2 = self.mffm2(f2, fg, fs)
        f1 = self.mffm1(f1, fg, fs)

        S_34, S_23, S_12 = self.decoder(f1, f2, f3, f4)

        return S_34, S_23, S_12


if __name__ == '__main__':
    import numpy as np
    from time import time

    net = Network(imagenet_pretrained=False)
    net.eval()

    dump_x = torch.randn(1, 3, 384, 384)
    frame_rate = np.zeros((1000, 1))
    for i in range(1000):
        start = time()
        y = net(dump_x)
        end = time()
        running_frame_rate = 1 * float(1 / (end - start))
        print(i, '->', running_frame_rate)
        frame_rate[i] = running_frame_rate
    print(np.mean(frame_rate))
    print(y.shape)