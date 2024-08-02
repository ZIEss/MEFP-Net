import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch


class Oconv_block(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Oconv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class conv_block(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
        self.ghost = GhostModule(out_ch)
        self.conv1 = nn.Sequential(
            nn.Conv2d(2 * out_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        x = self.ghost(x)
        x = self.conv1(x)
        return x


class GhostModule(nn.Module):
    def __init__(self, channel):
        super(GhostModule, self).__init__()
        self.oup = 2 * channel

        # 第一次卷积：得到通道数为init_channels，是输出的 1/ratio
        self.primary_conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True))

        # 第二次卷积：注意有个参数groups，为分组卷积
        # 每个feature map被卷积成 raito-1 个新的 feature map
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, groups=channel, bias=True),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out


class block(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class decode_block(nn.Module):
    def __init__(self, inc, dim, layer_scale_init_value=1e-6):
        super(decode_block, self).__init__()
        self.dga = DilatedGatedAttention(inc, dim)
        # self.conv1 = nn.Conv2d(inc, dim, kernel_size=3, stride=1, padding=1)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.cbma = CBAM(dim)

    def forward(self, x):
        x = self.dga(x)
        # x = self.conv1(x)
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        x = self.cbma(x)
        return x


class DilatedGatedAttention(nn.Module):
    def __init__(self, in_c, out_c, k_size=3, dilated_ratio=[9, 5, 2, 1]):
        super().__init__()

        self.mda0 = nn.Conv2d(in_c // 4, in_c // 4, kernel_size=k_size, stride=1,
                              padding=(k_size + (k_size - 1) * (dilated_ratio[0] - 1)) // 2,
                              dilation=dilated_ratio[0], groups=in_c // 4)
        self.mda1 = nn.Conv2d(in_c // 4, in_c // 4, kernel_size=k_size, stride=1,
                              padding=(k_size + (k_size - 1) * (dilated_ratio[1] - 1)) // 2,
                              dilation=dilated_ratio[1], groups=in_c // 4)
        self.mda2 = nn.Conv2d(in_c // 4, in_c // 4, kernel_size=k_size, stride=1,
                              padding=(k_size + (k_size - 1) * (dilated_ratio[2] - 1)) // 2,
                              dilation=dilated_ratio[2], groups=in_c // 4)
        self.mda3 = nn.Conv2d(in_c // 4, in_c // 4, kernel_size=k_size, stride=1,
                              padding=(k_size + (k_size - 1) * (dilated_ratio[3] - 1)) // 2,
                              dilation=dilated_ratio[3], groups=in_c // 4)
        self.norm_layer = nn.GroupNorm(4, in_c)
        self.norm_layer1 = nn.GroupNorm(4, out_c)
        self.conv = nn.Conv2d(in_c, out_c, 1)

    def forward(self, x):
        rex = self.conv(x)
        rex = self.norm_layer1(rex)
        x = torch.chunk(x, 4, dim=1)
        x0 = self.mda0(x[0])
        x1 = self.mda1(x[1])
        x2 = self.mda2(x[2])
        x3 = self.mda3(x[3])
        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = self.norm_layer(x)
        x = self.conv(x)
        x = F.gelu(x + rex)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 4, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 4, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


"""
    构造下采样模块--右边特征融合基础模块    
"""


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


"""
    模型主架构
"""


class Model(nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super(Model, self).__init__()

        # 卷积参数设置
        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        # 最大池化层
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 左边特征提取卷积层
        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = block(filters[3], filters[4])

        self.OConv1 = Oconv_block(in_ch, filters[0])
        self.OConv2 = Oconv_block(filters[0], filters[1])
        self.OConv3 = Oconv_block(filters[1], filters[2])
        self.OConv4 = Oconv_block(filters[2], filters[3])

        # 右边特征融合反卷积层
        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = decode_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = decode_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = decode_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = decode_block(filters[1], filters[0])

        self.as_pp = DenseASPP(filters[0], atrous_rates=[6, 12, 18])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        e1 = self.Conv1(x)
        ee1 = self.OConv1(x)
        e1 = e1 + ee1
        ee2 = self.Maxpool1(ee1)
        e2 = self.Maxpool1(e1)

        e2 = self.Conv2(e2)
        ee2 = self.OConv2(ee2)
        e2 = ee2 + e2
        ee3 = self.Maxpool2(ee2)
        e3 = self.Maxpool2(e2)

        e3 = self.Conv3(e3)
        ee3 = self.OConv3(ee3)
        e3 = ee3 + e3
        ee4 = self.Maxpool3(ee3)
        e4 = self.Maxpool3(e3)

        e4 = self.Conv4(e4)
        ee4 = self.OConv4(ee4)
        e4 = ee4 + e4
        ee5 = self.Maxpool4(ee4)
        e5 = self.Maxpool4(e4)

        e5 = self.Conv5(e5 + ee5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)  # 将e4特征图与d5特征图横向拼接
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)  # 将e3特征图与d4特征图横向拼接
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)  # 将e2特征图与d3特征图横向拼接
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)  # 将e1特征图与d1特征图横向拼接
        d2 = self.Up_conv2(d2)

        d2 = self.as_pp(d2)

        out = self.Conv(d2)

        return out


class DenseASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(DenseASPP, self).__init__()
        out_channels = in_channels
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.aspp_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=True),
            nn.GroupNorm(4, out_channels),
            nn.LeakyReLU(0.2, inplace=True))

        self.aspp_6 = ASPPConv(in_channels + 4, in_channels, rate1)
        self.aspp_12 = ASPPConv(2 * in_channels + 4, in_channels, rate2)
        self.aspp_18 = ASPPConv(3 * in_channels + 4, in_channels, rate3)
        self.aspp_pooling = ASPPPooling(in_channels, in_channels)
        self.ppm = PyramidPooling(in_channels)
        self.pp = nn.Sequential(
            nn.Conv2d(out_channels + 4, out_channels, 1, bias=True),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True))

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=True),
            nn.GroupNorm(4, out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1), )

    def forward(self, x):
        identity = x
        aspp_1 = self.aspp_1(x)
        ppm = self.ppm(x)
        x = torch.cat([x, ppm], dim=1)
        aspp6 = self.aspp_6(x)
        x = torch.cat([aspp6, x], dim=1)
        aspp12 = self.aspp_12(x)
        x = torch.cat([aspp12, x], dim=1)
        aspp18 = self.aspp_18(x)
        aspp_pooling = self.aspp_pooling(identity)
        aspp_pooling = self.pp(torch.cat([aspp_pooling, ppm], dim=1))
        x = torch.cat([aspp18, aspp12, aspp6, aspp_pooling, aspp_1], dim=1)
        return self.project(x)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(16),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class PyramidPooling(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels):
        super(PyramidPooling, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, 1, 1)
        self.conv3 = nn.Conv2d(in_channels, 1, 1)
        self.conv4 = nn.Conv2d(in_channels, 1, 1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)  # 自适应的平均池化，目标size分别为1x1,2x2,3x3,6x6
        return avgpool(x)

    def upsample(self, x, size):  # 上采样使用双线性插值
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 4)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([feat1, feat2, feat3, feat4], dim=1)  # concat 四个池化的结果
        return x
