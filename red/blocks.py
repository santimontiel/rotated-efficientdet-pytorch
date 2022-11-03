import torch.nn as nn

class ConvBlock(nn.Module):
    """ A convolutional block consisting of a 2D-convolution, batch norm,
    and activation function.
    """

    def __init__(self, in_chn, out_chn, kernel_size, stride=1,
                 padding=0, groups=1, bn=True, act=True) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_chn, out_chn, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_chn) if bn else nn.Identity()
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, i):
        x = self.conv(i)
        x = self.bn(x)
        x = self.act(x)
        return x


class MBConvN(nn.Module):
    """ MobileNet-V2 Inverted residual block. Also referred as 'MBConv'
    at EfficientNet.
    """

    def __init__(
            self, in_chn, out_chn, kernel_size, stride=1, expansion_factor=1,
            do_se = True
        ) -> None:
        super().__init__()

        if stride not in [1, 2]:
            raise ValueError(f"Stride is {stride}, but must be 1 or 2.")

        padding = (kernel_size-1 // 2)
        expand_chn = expansion_factor * in_chn
        self.do_skip = (stride == 1) and (in_chn == out_chn)
        self.do_se = do_se

        # NOTE: A 1x1 conv with in_chn=out_chn and kernel_size=1 is equivalent
        # to identity operation in matrices.
        self.conv_pw1 = nn.Identity() if (expansion_factor == 1) else ConvBlock(
            in_chn, expand_chn, kernel_size=1)
        self.conv_dw = ConvBlock(expand_chn, expand_chn, kernel_size, stride,
            padding, groups=expand_chn)
        self.se_block = SEBlock(expand_chn, r=4) if (self.do_se) else nn.Identity()
        self.conv_pw2 = ConvBlock(expand_chn, out_chn, 1, stride, act=False)

    def forward(self, x):
        
        res = x
        x = self.conv_pw1(x)
        x = self.conv_dw(x)
        x = self.se_block(x)
        x = self.conv_pw2(x)
        if self.do_skip:
            x = x + res
        return x


class FusedMBConvN(nn.Module):

    def __init__(
            self, in_chn, out_chn, kernel_size, stride, expansion_factor=1,
            do_se=True
        ) -> None:
        super().__init__()

        if stride not in [1, 2]:
            raise ValueError(f"Stride is {stride}. It must be 1 or 2.")
        
        padding = (kernel_size-1 // 2)
        expand_chn = in_chn * expansion_factor
        self.do_skip = (stride == 1) and (in_chn == out_chn)
        self.do_se = do_se

        self.conv_fused = ConvBlock(in_chn, expand_chn, 3, stride, padding)
        self.se_block = SEBlock(expand_chn, r=4) if self.do_se else nn.Identity()
        self.reduce_pw = ConvBlock(expand_chn, out_chn, 1, stride, act=False)

    def forward(self, x):
        res = x
        x = self.conv_fused(x)
        x = self.se_block(x)
        x = self.reduce_pw(x)
        if self.do_skip:
            x = x + res
        return x


class SEBlock(nn.Module):

    def __init__(self, c, r=4) -> None:
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.SiLU(),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, i):
        bs, c, _, _ = i.shape
        x = self.squeeze(i).view(bs, c)
        x = self.excitation(x).view(bs, c, 1, 1)
        return i * x.expand_as(i)