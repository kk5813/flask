import torch
import torch.nn as nn


# 这是一个上采样模块，用于U-Net的解码器部分。
# 包含两个卷积层（conv1和conv2）和一个双线性上采样层（up）
class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


class Unet(nn.Module):
    def __init__(self, num_classes=4, pretrained=False, backbone='vgg'):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg = VGG16(pretrained=pretrained)
            in_filters = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained=pretrained)
            in_filters = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]

        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

        self.sknet1 = SKAttention()
        self.ela = ELA(in_channels=512, phi='B')

    def forward(self, inputs):
        # print("Input size:", inputs.size())  # 打印输入尺寸
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        up4 = self.up_concat4(feat4, feat5)
        up4 = self.sknet1(up4)
        up4 = self.ela(up4)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)

        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True


import math

import torch.nn as nn
import torch.utils.model_zoo as model_zoo


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # 利用1x1卷积下降通道数
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        # 利用3x3卷积进行特征提取
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        # 利用1x1卷积上升通道数
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        # -----------------------------------------------------------#
        #   假设输入图像为600,600,3
        #   当我们使用resnet50的时候
        # -----------------------------------------------------------#
        self.inplanes = 64
        super(ResNet, self).__init__()
        # 600,600,3 -> 300,300,64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # 300,300,64 -> 150,150,64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)  # change
        # 150,150,64 -> 150,150,256
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 150,150,256 -> 75,75,512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 75,75,512 -> 38,38,1024
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # 38,38,1024 -> 19,19,2048
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        x = self.conv1(x)
        x = self.bn1(x)
        feat1 = self.relu(x)

        x = self.maxpool(feat1)
        feat2 = self.layer1(x)

        feat3 = self.layer2(feat2)
        feat4 = self.layer3(feat3)
        feat5 = self.layer4(feat4)
        return [feat1, feat2, feat3, feat4, feat5]


def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(
            model_zoo.load_url('https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth', model_dir='model_data'),
            strict=False)

    del model.avgpool
    del model.fc
    return model


import torch.nn as nn
from torch.hub import load_state_dict_from_url


class VGG(nn.Module):
    def __init__(self, features, num_classes=4):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        # x = self.features(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        feat1 = self.features[:4](x)
        feat2 = self.features[4:9](feat1)
        feat3 = self.features[9:16](feat2)
        feat4 = self.features[16:23](feat3)
        feat5 = self.features[23:-1](feat4)
        return [feat1, feat2, feat3, feat4, feat5]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False, in_channels=3):
    layers = []
    for v in cfg:
        if v == 'M':  # 最大池化
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# 512,512,3 -> 512,512,64 -> 256,256,64 -> 256,256,128 -> 128,128,128 -> 128,128,256 -> 64,64,256
# 64,64,512 -> 32,32,512 -> 32,32,512
cfgs = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}


def VGG16(pretrained, in_channels=3, **kwargs):
    model = VGG(make_layers(cfgs["D"], batch_norm=False, in_channels=in_channels), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth",
                                              model_dir="./model_data")
        model.load_state_dict(state_dict)

    del model.avgpool
    del model.classifier
    return model


import torch
import torch.nn as nn


class ELA(nn.Module):
    def __init__(self, in_channels, phi):
        super(ELA, self).__init__()
        """
        ELA-T 和 ELA-B 设计为轻量级，非常适合网络层数较少或轻量级网络的 CNN 架构
        ELA-B 和 ELA-S 在具有更深结构的网络上表现最佳
        ELA-L 特别适合大型网络。

        参数:
        - in_channels (int): 输入特征图的通道数
        - phi (str): 表示卷积核大小和组数的选择，'T', 'B', 'S', 'L'中的一个
        """
        # 根据 phi 参数选择不同的卷积核大小
        Kernel_size = {'T': 5, 'B': 7, 'S': 5, 'L': 7}[phi]
        # 根据 phi 参数选择不同的卷积组数
        groups = {'T': in_channels, 'B': in_channels, 'S': in_channels // 8, 'L': in_channels // 8}[phi]
        # 根据 phi 参数选择不同的归一化组数
        num_groups = {'T': 32, 'B': 16, 'S': 16, 'L': 16}[phi]
        # 计算填充大小以保持卷积后尺寸不变
        pad = Kernel_size // 2
        # 1D 卷积层，使用分组卷积，卷积核大小为 Kernel_size
        self.con1 = nn.Conv1d(in_channels, in_channels, kernel_size=Kernel_size, padding=pad, groups=groups, bias=False)
        # 组归一化层
        self.GN = nn.GroupNorm(num_groups, in_channels)
        # Sigmoid 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        """
        前向传播函数。
        参数:
        - input (torch.Tensor): 输入特征图，形状为 (batch_size, channels, height, width)
        返回:
        - torch.Tensor: 应用边缘注意力后的特征图
        """
        b, c, h, w = input.size()  # 获取输入特征图的形状
        # 在宽度方向上进行平均池化
        x_h = torch.mean(input, dim=3, keepdim=True).view(b, c, h)
        # 在高度方向上进行平均池化
        x_w = torch.mean(input, dim=2, keepdim=True).view(b, c, w)
        # 对池化后的特征图应用 1D 卷积
        x_h = self.con1(x_h)  # [b, c, h]
        x_w = self.con1(x_w)  # [b, c, w]
        # 对卷积后的特征图进行归一化和激活，并 reshape 回来
        x_h = self.sigmoid(self.GN(x_h)).view(b, c, h, 1)  # [b, c, h, 1]
        x_w = self.sigmoid(self.GN(x_w)).view(b, c, 1, w)  # [b, c, 1, w]
        # 将输入特征图、x_h 和 x_w 按元素相乘，得到最终的输出特征图
        return x_h * x_w * input


if __name__ == "__main__":
    # 创建一个形状为 [batch_size, channels, height, width] 的虚拟输入张量
    input = torch.randn(1, 32, 256, 256)
    ela = ELA(in_channels=32, phi='T')
    output = ela(input)
    print(output.size())

import numpy as np
import torch
from torch import nn
from torch.nn import init
from collections import OrderedDict


class SKAttention(nn.Module):
    def __init__(self, channel=512, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32):
        """
        SKAttention 初始化函数
        Args:
            channel (int): 输入和输出通道数。
            kernels (list): 多尺度卷积核的大小列表。
            reduction (int): 通道数缩减的比例因子。
            group (int): 深度卷积的组数。
            L (int): 计算降维的最小通道数。
        """
        super().__init__()

        # 计算缩减后的通道数，保证其不小于 L。
        self.d = max(L, channel // reduction)

        # 初始化多个卷积操作，每个卷积操作的卷积核大小由 kernels 列表决定。
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2, groups=group)),  # 深度卷积
                    ('bn', nn.BatchNorm2d(channel)),  # 批归一化
                    ('relu', nn.ReLU())  # ReLU 激活函数
                ]))
            )

        # 线性层，用于将通道数降维为 d。
        self.fc = nn.Linear(channel, self.d)

        # 初始化多个线性层，用于将降维后的特征映射回原通道数。
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))

        # softmax 层，用于计算不同尺度特征的注意力权重。
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        """
        前向传播函数
        Args:
            x (Tensor): 输入张量，形状为 (batch_size, channels, height, width)。
        Returns:
            Tensor: 输出张量，形状与输入相同。
        """
        bs, c, _, _ = x.size()  # 获取输入张量的形状信息
        conv_outs = []

        ### 多尺度特征提取
        for conv in self.convs:
            conv_outs.append(conv(x))  # 使用不同的卷积核对输入进行卷积操作
        feats = torch.stack(conv_outs, 0)  # 将不同卷积核的输出在第一个维度上堆叠，形状为 (k, bs, channel, h, w)

        ### 特征融合
        U = sum(conv_outs)  # 将所有尺度的特征进行相加，形状为 (bs, c, h, w)

        ### 通道数缩减
        S = U.mean(-1).mean(-1)  # 对空间维度进行平均，得到形状为 (bs, c) 的张量
        Z = self.fc(S)  # 通过全连接层进行通道数缩减，得到形状为 (bs, d) 的张量

        ### 计算注意力权重
        weights = []
        for fc in self.fcs:
            weight = fc(Z)  # 通过线性层将降维后的特征映射回原通道数，形状为 (bs, c)
            weights.append(weight.view(bs, c, 1, 1))  # 调整形状为 (bs, channel, 1, 1)
        attention_weights = torch.stack(weights, 0)  # 将所有的注意力权重在第一个维度上堆叠，形状为 (k, bs, channel, 1, 1)
        attention_weights = self.softmax(attention_weights)  # 使用 softmax 进行归一化，得到最终的注意力权重

        ### 加权融合特征
        V = (attention_weights * feats).sum(0)  # 将注意力权重与对应的多尺度特征相乘并相加，得到最终的加权特征
        return V


import torch
import torch.nn as nn
from einops import rearrange  # 导入 rearrange 函数，用于重排张量


# 定义简化版的线性注意力类，继承自 nn.Module
class SimplifiedLinearAttention(nn.Module):
    r"""
    参数:
        dim (int): 输入通道数。
        window_size (tuple[int]): 窗口的高和宽。
        num_heads (int): 注意力头的数量。
        qkv_bias (bool, 可选): 如果为 True，则为查询、键和值添加一个可学习的偏置。默认值：True
        qk_scale (float | None, 可选): 如果设置，则覆盖默认的 qk scale，即 head_dim ** -0.5。
        attn_drop (float, 可选): 注意力权重的丢弃比率。默认值：0.0
        proj_drop (float, 可选): 输出的丢弃比率。默认值：0.0
    """

    # 初始化函数，定义了注意力模块的基本结构
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 focusing_factor=3, kernel_size=5):

        super().__init__()
        self.dim = dim  # 输入通道数
        self.window_size = window_size  # 窗口大小，分别为高和宽
        self.num_heads = num_heads  # 注意力头数
        head_dim = dim // num_heads  # 每个注意力头的维度

        self.focusing_factor = focusing_factor  # 聚焦因子
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 线性变换，用于生成查询、键和值
        self.attn_drop = nn.Dropout(attn_drop)  # 注意力权重的 dropout
        self.proj = nn.Linear(dim, dim)  # 最终的线性变换
        self.proj_drop = nn.Dropout(proj_drop)  # 输出的 dropout

        self.softmax = nn.Softmax(dim=-1)  # 定义 Softmax 函数用于计算权重

        # 深度可分离卷积 (Depthwise Convolution)，用于捕捉局部特征
        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)

        # 可学习的相对位置编码
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, window_size[0] * window_size[1], dim)))

        # 打印初始化信息，便于调试
        print('Linear Attention window{} f{} kernel{}'.format(window_size, focusing_factor, kernel_size))

    # 前向传播函数
    def forward(self, x, mask=None):
        """
        参数:
            x: 输入特征，形状为 (num_windows*B, N, C)
            mask: (0/-inf) 掩码，形状为 (num_windows, Wh*Ww, Wh*Ww) 或 None
        """
        B, N, C = x.shape

        # # 获取输入的形状信息
        # B, C ,H ,W = x.shape  # B是批次大小，C是通道数，H和W是高度和宽度
        # N = H*W  # N是图像的总像素数
        # x = x.view(B,N,C)  # 调整输入形状为 (B, N, C)

        # 对 x 进行线性变换，并重新排列得到查询、键和值
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv.unbind(0)  # 将查询、键和值分开
        k = k + self.positional_encoding  # 为键添加相对位置编码

        kernel_function = nn.ReLU()  # 使用 ReLU 作为激活函数
        q = kernel_function(q)
        k = kernel_function(k)

        # 对查询、键和值进行多头分解
        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]  # 获取形状信息

        # 使用混合精度训练，保证数值精度并优化计算效率
        with torch.cuda.amp.autocast(enabled=False):
            q = q.to(torch.float32)
            k = k.to(torch.float32)
            v = v.to(torch.float32)

            # 计算注意力权重
            z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
            if i * j * (c + d) > c * d * (i + j):  # 根据输入规模选择不同的计算路径
                kv = torch.einsum("b j c, b j d -> b c d", k, v)  # 计算键值对的内积
                x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)  # 结合查询计算最终结果
            else:
                qk = torch.einsum("b i c, b j c -> b i j", q, k)  # 查询和键的点积
                x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)  # 结合查询计算输出

        # 将特征图恢复为二维形状
        num = int(v.shape[1] ** 0.5)
        feature_map = rearrange(v, "b (w h) c -> b c w h", w=num, h=num)  # 重排为 (B, C, W, H)
        feature_map = rearrange(self.dwc(feature_map), "b c w h -> b (w h) c")  # 经过卷积后重排回 (B, N, C)
        x = x + feature_map  # 将卷积结果与注意力结果相加

        # 恢复多头后的形状
        x = rearrange(x, "(b h) n c -> b n (h c)", h=self.num_heads)
        x = self.proj(x)  # 线性变换
        x = self.proj_drop(x)  # 输出 dropout
        # print(x.shape) # torch.Size([4, 1024, 64])
        # x = x.permute(0,2,1).view(B,C,H,W)  # 恢复输入的原始形状

        return x  # 返回最终输出


import torch
import torch.nn as nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        """
        初始化SELayer类。
        参数:
        channel (int): 输入特征图的通道数。
        reduction (int): 用于减少通道数的缩减率，默认为16。它用于在全连接层中压缩特征的维度。
        """
        super(SELayer, self).__init__()
        # 自适应平均池化层，将每个通道的空间维度（H, W）压缩到1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 全连接层序列，包含两个线性变换和中间的ReLU激活函数
        self.fc = nn.Sequential(
            # 第一个线性层，将通道数从 'channel' 缩减到 'channel // reduction'
            nn.Linear(channel, channel // reduction, bias=False),
            # ReLU激活函数，用于引入非线性
            nn.ReLU(inplace=True),
            # 第二个线性层，将通道数从 'channel // reduction' 恢复到 'channel'
            nn.Linear(channel // reduction, channel, bias=False),
            # Sigmoid激活函数，将输出限制在(0, 1)之间
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        前向传播函数。
        参数:
        x (Tensor): 输入张量，形状为 (batch_size, channel, height, width)。
        返回:
        Tensor: 经过通道注意力调整后的输出张量，形状与输入相同。
        """
        # 获取输入张量的形状
        b, c, h, w = x.size()
        # Squeeze：通过全局平均池化层，将每个通道的空间维度（H, W）压缩到1x1
        y = self.avg_pool(x).view(b, c)
        # Excitation：通过全连接层序列，对压缩后的特征进行处理
        y = self.fc(y).view(b, c, 1, 1)
        # 通过扩展后的注意力权重 y 调整输入张量 x 的每个通道
        return x * y.expand_as(x)
