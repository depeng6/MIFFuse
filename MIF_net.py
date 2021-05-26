# -*- encoding: utf-8 -*-
"""
@File    : MIF_net.py
@Time    : 2021/3/10 15:48
@Author  : Chool
@Email   : depeng_cust@163.com
@Software: PyCharm
"""
import torch.nn as nn
import torch

def conv3x3(in_planes,out_planes,stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes,out_planes,kernel_size=3,stride=stride,
                     padding=1,bias=False)


class BaseBlock(nn.Module):
    def __init__(self,in_planes,out_planes,stride=1):
        super(BaseBlock,self).__init__()
        self.conv1 = conv3x3(in_planes,out_planes,stride)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.sigmoid = nn.Sigmoid()
        self.conv2 = conv3x3(out_planes,out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        if in_planes != out_planes:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes,out_planes,kernel_size=1,
                                                      stride=stride,bias=False))
        else:
            self.downsample = lambda x: x


    def forward(self,x):
        residula = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sigmoid(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residula
        out = self.sigmoid(out)
        return out

class PreActBasicBlock(BaseBlock):
    def __init__(self,in_planes,out_planes,stride):
        super(PreActBasicBlock,self).__init__(in_planes,out_planes,stride)
        if in_planes != out_planes:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes,out_planes,kernel_size=1,
                                                      stride=stride,bias=False))
        else:
            self.downsample = lambda x:x

    def forward(self,x):
        res = self.downsample(x)
        out = self.sigmoid(x)
        out = self.conv1(out)

        out = self.sigmoid(out)
        out = self.conv2(out)

        out += res
        return out


# Convolution operation
class Conv_block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,is_last=False):
        super(Conv_block, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride,padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.is_last = is_last
    def forward(self, input):
        out = self.conv2d(input)
        if self.is_last:
            return out
        else:
            out = self.relu(out)
            return out


class FusionNet(nn.Module):
    def __init__(self,in_planes,out_planes):
        super(FusionNet,self).__init__()
        kernel_size = [1,3]
        filter_n = [16,32,64]

        self.conv1_1 = Conv_block(in_planes,filter_n[0],kernel_size=3,stride=1,padding=1)
        self.conv1_2 = Conv_block(in_planes,filter_n[0],kernel_size=3,stride=1,padding=1)

        self.conv2_1 = PreActBasicBlock(filter_n[0],filter_n[1],stride=1)
        self.conv2_2 = PreActBasicBlock(filter_n[0],filter_n[1],stride=1)

        self.conv_cat2 = Conv_block(filter_n[2],filter_n[1],kernel_size=1,stride=1,padding=0)

        self.conv3_1 = PreActBasicBlock(filter_n[2],filter_n[1],stride=1)
        self.conv3_2 = PreActBasicBlock(filter_n[2],filter_n[1],stride=1)

        self.conv_cat3 = Conv_block(filter_n[2], filter_n[1], kernel_size=1, stride=1, padding=0)

        self.conv4_1 = PreActBasicBlock(filter_n[2],filter_n[1],stride=1)
        self.conv4_2 = PreActBasicBlock(filter_n[2],filter_n[1],stride=1)

        self.conv_cat4 = Conv_block(filter_n[2], filter_n[1], kernel_size=1, stride=1, padding=0)

        self.conv5_1 = PreActBasicBlock(filter_n[2],filter_n[1],stride=1)
        self.conv5_2 = PreActBasicBlock(filter_n[2],filter_n[1],stride=1)

        self.conv_cat5 = Conv_block(filter_n[2], filter_n[1], kernel_size=1, stride=1, padding=0)
        self.conv_cat35_1 = Conv_block(filter_n[2], filter_n[1], kernel_size=1, stride=1, padding=0)
        self.conv_cat35_2 = Conv_block(filter_n[2], filter_n[1], kernel_size=1, stride=1, padding=0)

        self.conv6_1 = PreActBasicBlock(filter_n[2],filter_n[1],stride=1)
        self.conv6_2 = PreActBasicBlock(filter_n[2],filter_n[1],stride=1)
        self.conv_cat26_1 = Conv_block(filter_n[2], filter_n[1], kernel_size=1, stride=1, padding=0)
        self.conv_cat26_2 = Conv_block(filter_n[2], filter_n[1], kernel_size=1, stride=1, padding=0)

        self.conv7 = Conv_block(filter_n[2],out_planes,kernel_size=3,stride=1,padding=1,is_last=True)


    def forward(self,input_ir,input_vi):
        x1_ir = self.conv1_1(input_ir)
        x1_vi = self.conv1_2(input_vi)

        x2_ir = self.conv2_1(x1_ir)
        x2_vi = self.conv2_2(x1_vi)

        x2_cat = torch.cat((x2_ir,x2_vi),1)
        x2_cat_conv = self.conv_cat2(x2_cat)

        x23_ir = torch.cat((x2_ir,x2_cat_conv),1)
        x23_vi = torch.cat((x2_vi,x2_cat_conv),1)

        x3_ir = self.conv3_1(x23_ir)
        x3_vi = self.conv3_2(x23_vi)

        x3_cat = torch.cat((x3_ir,x3_vi),1)
        x3_cat_conv = self.conv_cat3(x3_cat)

        x34_ir = torch.cat((x3_ir,x3_cat_conv),1)
        x34_vi = torch.cat((x3_vi,x3_cat_conv),1)

        x4_ir = self.conv4_1(x34_ir)
        x4_vi = self.conv4_2(x34_vi)

        x4_cat = torch.cat((x4_ir,x4_vi),1)
        x4_cat_conv = self.conv_cat4(x4_cat)

        x45_ir = torch.cat((x4_ir,x4_cat_conv),1)
        x45_vi = torch.cat((x4_vi,x4_cat_conv),1)

        x5_ir = self.conv5_1(x45_ir)
        x5_vi = self.conv5_2(x45_vi)

        x35_cat_ir = torch.cat((x3_ir,x5_ir),1)
        x35_cat_conv_ir = self.conv_cat35_1(x35_cat_ir)
        x35_cat_vi = torch.cat((x3_vi,x5_vi),1)
        x35_cat_conv_vi = self.conv_cat35_2(x35_cat_vi)

        x5_cat = torch.cat((x35_cat_conv_ir,x35_cat_conv_vi),1)
        x5_cat_conv = self.conv_cat5(x5_cat)

        x56_ir = torch.cat((x35_cat_conv_ir,x5_cat_conv),1)
        x56_vi = torch.cat((x35_cat_conv_vi,x5_cat_conv),1)

        x6_ir = self.conv6_1(x56_ir)
        x6_vi = self.conv6_2(x56_vi)

        x26_cat_ir = torch.cat((x2_ir,x6_ir),1)
        x26_cat_conv_ir = self.conv_cat26_1(x26_cat_ir)
        x26_cat_vi = torch.cat((x2_vi,x6_vi),1)
        x26_cat_conv_vi = self.conv_cat26_2(x26_cat_vi)

        x6_cat = torch.cat((x26_cat_conv_ir,x26_cat_conv_vi),1)
        x7 = self.conv7(x6_cat)

        return x7

if __name__ == '__main__':
    image1 = torch.rand(32,3,80,80)
    image2 = torch.rand(32,3,80,80)

    fuse = FusionNet(in_planes=3,out_planes=1)
    out1 = fuse(image1,image2)

    print(out1.shape)
    # print(out2.shape)














