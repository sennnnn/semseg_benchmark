import torch
import torch.nn as nn
import torch.nn.functional as F

from sswss.models.backbones.base_net import BaseNet


class ResBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 stride=1,
                 first_dilation=None,
                 dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(ResBlock, self).__init__()

        self.same_shape = (in_channels == out_channels and stride == 1)

        if first_dilation is None:
            first_dilation = dilation

        self.bn_branch2a = norm_layer(in_channels)

        self.conv_branch2a = nn.Conv2d(
            in_channels, mid_channels, 3, stride, padding=first_dilation, dilation=first_dilation, bias=False)

        self.bn_branch2b1 = norm_layer(mid_channels)

        self.conv_branch2b1 = nn.Conv2d(mid_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False)

        if not self.same_shape:
            self.conv_branch1 = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)

    def forward(self, x, get_x_bn_relu=False):

        branch2 = self.bn_branch2a(x)
        branch2 = F.relu(branch2)

        x_bn_relu = branch2

        if not self.same_shape:
            branch1 = self.conv_branch1(branch2)
        else:
            branch1 = x

        branch2 = self.conv_branch2a(branch2)
        branch2 = self.bn_branch2b1(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.conv_branch2b1(branch2)

        x = branch1 + branch2

        if get_x_bn_relu:
            return x, x_bn_relu

        return x

    def __call__(self, x, get_x_bn_relu=False):
        return self.forward(x, get_x_bn_relu=get_x_bn_relu)


class ResBlock_bot(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, dilation=1, dropout=0., norm_layer=nn.BatchNorm2d):
        super(ResBlock_bot, self).__init__()

        self.same_shape = (in_channels == out_channels and stride == 1)

        self.bn_branch2a = norm_layer(in_channels)
        self.conv_branch2a = nn.Conv2d(in_channels, out_channels // 4, 1, stride, bias=False)

        self.bn_branch2b1 = norm_layer(out_channels // 4)
        self.dropout_2b1 = torch.nn.Dropout2d(dropout)
        self.conv_branch2b1 = nn.Conv2d(
            out_channels // 4, out_channels // 2, 3, padding=dilation, dilation=dilation, bias=False)

        self.bn_branch2b2 = norm_layer(out_channels // 2)
        self.dropout_2b2 = torch.nn.Dropout2d(dropout)
        self.conv_branch2b2 = nn.Conv2d(out_channels // 2, out_channels, 1, bias=False)

        if not self.same_shape:
            self.conv_branch1 = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)

    def forward(self, x, get_x_bn_relu=False):

        branch2 = self.bn_branch2a(x)
        branch2 = F.relu(branch2)
        x_bn_relu = branch2

        branch1 = self.conv_branch1(branch2)

        branch2 = self.conv_branch2a(branch2)

        branch2 = self.bn_branch2b1(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.dropout_2b1(branch2)
        branch2 = self.conv_branch2b1(branch2)

        branch2 = self.bn_branch2b2(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.dropout_2b2(branch2)
        branch2 = self.conv_branch2b2(branch2)

        x = branch1 + branch2

        if get_x_bn_relu:
            return x, x_bn_relu

        return x

    def __call__(self, x, get_x_bn_relu=False):
        return self.forward(x, get_x_bn_relu=get_x_bn_relu)


class ResNet38d(BaseNet):

    out_channels = [128, 256, 512, 1024, 4096]

    def __init__(self, norm_layer):
        super(ResNet38d, self).__init__()

        self.conv1a = nn.Conv2d(3, 64, 3, padding=1, bias=False)

        self.b2 = ResBlock(64, 128, 128, stride=2, norm_layer=norm_layer)
        self.b2_1 = ResBlock(128, 128, 128, norm_layer=norm_layer)
        self.b2_2 = ResBlock(128, 128, 128, norm_layer=norm_layer)

        self.b3 = ResBlock(128, 256, 256, stride=2, norm_layer=norm_layer)
        self.b3_1 = ResBlock(256, 256, 256, norm_layer=norm_layer)
        self.b3_2 = ResBlock(256, 256, 256, norm_layer=norm_layer)

        self.b4 = ResBlock(256, 512, 512, stride=2, norm_layer=norm_layer)
        self.b4_1 = ResBlock(512, 512, 512, norm_layer=norm_layer)
        self.b4_2 = ResBlock(512, 512, 512, norm_layer=norm_layer)
        self.b4_3 = ResBlock(512, 512, 512, norm_layer=norm_layer)
        self.b4_4 = ResBlock(512, 512, 512, norm_layer=norm_layer)
        self.b4_5 = ResBlock(512, 512, 512, norm_layer=norm_layer)

        self.b5 = ResBlock(512, 512, 1024, stride=1, first_dilation=1, dilation=2)
        self.b5_1 = ResBlock(1024, 512, 1024, dilation=2)
        self.b5_2 = ResBlock(1024, 512, 1024, dilation=2)

        self.b6 = ResBlock_bot(1024, 2048, stride=1, dilation=4, dropout=0.3, norm_layer=norm_layer)
        self.b7 = ResBlock_bot(2048, 4096, dilation=4, dropout=0.5, norm_layer=norm_layer)
        self.bn7 = norm_layer(4096)

        # fixing the parameters
        self._fix_params([self.conv1a, self.b2, self.b2_1, self.b2_2])

    def fan_out(self):
        return 4096

    def forward(self, x):
        return self.forward_as_dict(x)['conv6']

    def forward_as_dict(self, x):

        x = self.conv1a(x)

        x = self.b2(x)
        x = self.b2_1(x)
        x = self.b2_2(x)
        conv2 = x

        x = self.b3(x)
        x = self.b3_1(x)
        x = self.b3_2(x)
        conv3 = x

        x = self.b4(x)
        x = self.b4_1(x)
        x = self.b4_2(x)
        x = self.b4_3(x)
        x = self.b4_4(x)
        x = self.b4_5(x)

        x, conv4 = self.b5(x, get_x_bn_relu=True)
        x = self.b5_1(x)
        x = self.b5_2(x)

        x, conv5 = self.b6(x, get_x_bn_relu=True)

        x = self.b7(x)
        conv6 = F.relu(self.bn7(x))

        return dict({'conv2': conv2, 'conv3': conv3, 'conv4': conv4, 'conv5': conv5, 'conv6': conv6})
