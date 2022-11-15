import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import ConvBlock

class DepthwiseConvBlock(nn.Module):
    """ Depthwise seperable convolution block with batch normalization and
    activation function.
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, bn=True, act=True):
        super(DepthwiseConvBlock, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, 
                               padding, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                   stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        self.act = nn.SiLU() if act else nn.Identity()
        
    def forward(self, inputs):
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class BiFPNBlock(nn.Module):

    def __init__(self, feature_size=64) -> None:
        super(BiFPNBlock, self).__init__()
        self.conv_p3_td = DepthwiseConvBlock(feature_size, feature_size)
        self.conv_p4_td = DepthwiseConvBlock(feature_size, feature_size)
        self.conv_p5_td = DepthwiseConvBlock(feature_size, feature_size)
        self.conv_p6_td = DepthwiseConvBlock(feature_size, feature_size)

        self.conv_p4_out = DepthwiseConvBlock(feature_size, feature_size)
        self.conv_p5_out = DepthwiseConvBlock(feature_size, feature_size)
        self.conv_p6_out = DepthwiseConvBlock(feature_size, feature_size)
        self.conv_p7_out = DepthwiseConvBlock(feature_size, feature_size)

        self.w_td = nn.Parameter(torch.Tensor(2,4))
        self.w_td_act = nn.SiLU()
        self.w_out = nn.Parameter(torch.Tensor(3,4))
        self.w_out_act = nn.SiLU()

    def forward(self, x):

        p3_in, p4_in, p5_in, p6_in, p7_in = x

        w_td = self.w_td_act(self.w_td)
        w_out = self.w_out_act(self.w_out)

        p7_td = p7_in
        p6_td = self.conv_p6_td(w_td[0,0] * p6_in + w_td[1,0] * F.interpolate(p7_in, scale_factor=2) / (w_td[0,0] + w_td[1,0] + 1e-6))
        p5_td = self.conv_p5_td(w_td[0,1] * p5_in + w_td[1,1] * F.interpolate(p6_in, scale_factor=2) / (w_td[0,1] + w_td[1,1] + 1e-6))
        p4_td = self.conv_p4_td(w_td[0,2] * p4_in + w_td[1,2] * F.interpolate(p5_in, scale_factor=2) / (w_td[0,2] + w_td[1,2] + 1e-6))
        p3_td = self.conv_p3_td(w_td[0,3] * p3_in + w_td[1,3] * F.interpolate(p4_in, scale_factor=2) / (w_td[0,3] + w_td[1,3] + 1e-6))
        
        p7_out = self.conv_p7_out(w_out[0,0] * p7_in + w_out[1,0] * p7_td + w_out[2,0] * nn.Upsample(scale_factor=0.5)(p6_out) / (w_out[0,0] + w_out[1,0] + w_out[2,0] + 1e-6))
        p6_out = self.conv_p6_out(w_out[0,1] * p6_in + w_out[1,1] * p6_td + w_out[2,1] * nn.Upsample(scale_factor=0.5)(p5_out) / (w_out[0,1] + w_out[1,1] + w_out[2,1] + 1e-6))
        p5_out = self.conv_p5_out(w_out[0,2] * p5_in + w_out[1,2] * p5_td + w_out[2,2] * nn.Upsample(scale_factor=0.5)(p4_out) / (w_out[0,2] + w_out[1,2] + w_out[2,2] + 1e-6))
        p4_out = self.conv_p4_out(w_out[0,3] * p4_in + w_out[1,3] * p4_td + w_out[2,3] * nn.Upsample(scale_factor=0.5)(p3_out) / (w_out[0,3] + w_out[1,3] + w_out[2,3] + 1e-6))
        p3_out = p3_td

        return [p3_out, p4_out, p5_out, p6_out, p7_out]


class BiFPN(nn.Module):

    def __init__(self, size, feature_size=64, num_layers=3) -> None:
        super(BiFPN, self).__init__()

        self.conv_p3 = nn.Conv2d()
        self.conv_p4 = nn.Conv2d()
        self.conv_p5 = nn.Conv2d()
        self.conv_p6 = nn.Conv2d()
        self.conv_p7 = ConvBlock(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        bifpns = []
        for _ in range(num_layers):
            bifpns.append(BiFPNBlock(feature_size))
        self.bifpn = nn.Sequential(*bifpns)

    def forward(self, x):
        c3, c4, c5 = x
        p3_in = self.conv_p3(c3)
        p4_in = self.conv_p4(c4)
        p5_in = self.conv_p5(c5)
        p6_in = self.conv_p6(c5)
        p7_in = self.conv_p7(p6_in)
        return [p3_in, p4_in, p5_in, p6_in, p7_in]

