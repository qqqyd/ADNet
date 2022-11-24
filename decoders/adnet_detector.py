from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
BatchNorm2d = nn.BatchNorm2d

class SD_Module(nn.Module):
    def __init__(self, in_dim):
        super(SD_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out


class SRM(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(SRM, self).__init__()
        inter_channels = in_channels
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.sd = SD_Module(inter_channels)
        self.si = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(inter_channels, out_channels, 3, padding=1, bias=False),
                                    norm_layer(out_channels),
                                    nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels, out_channels, 1), norm_layer(out_channels),
                                   nn.ReLU())

    def forward(self, x):
        feat1 = self.conv1(x)
        sd_feat = self.sd(feat1)
        sd_feat = self.conv2(sd_feat)

        si_feat_att = self.si(x)
        si_feat_att = torch.sigmoid(si_feat_att)
        si_feat = x * si_feat_att + x
        spm_output = sd_feat + si_feat
        spm_output = self.conv3(spm_output)
        return spm_output


class ADNetDetector(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 inner_channels=256,
                 bias=False,
                 *args, **kwargs):
        super(ADNetDetector, self).__init__()
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')

        self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1, bias=bias)
        self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1, bias=bias)
        self.in3 = nn.Conv2d(in_channels[-3], inner_channels, 1, bias=bias)
        self.in2 = nn.Conv2d(in_channels[-4], inner_channels, 1, bias=bias)

        self.out5 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=8, mode='nearest'))
        self.out4 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=4, mode='nearest'))
        self.out3 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.out2 = nn.Conv2d(
            inner_channels, inner_channels//4, 3, padding=1, bias=bias)

        self.shrink_pred1 = self._init_pred1(inner_channels, bias)
        self.origin_pred1 = self._init_pred1(inner_channels, bias)
        self.shrink_pred2 = self._init_pred2(inner_channels, bias)
        self.origin_pred2 = self._init_pred2(inner_channels, bias)
        self.dilate_pred = nn.Sequential(
            nn.Conv2d(inner_channels // 2, inner_channels //
                      4, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, inner_channels // 4, 2, 2),
            BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, 1, 2, 2))
        self.get_atten = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels // 4, 1, 3, padding=1, bias=bias))
        self.up_trans = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels, inner_channels, 2, 2))
        self.avgpool = nn.AvgPool2d(2)
        self.srm = SRM(inner_channels, inner_channels, BatchNorm2d)

        self.in5.apply(self.weights_init)
        self.in4.apply(self.weights_init)
        self.in3.apply(self.weights_init)
        self.in2.apply(self.weights_init)
        self.out5.apply(self.weights_init)
        self.out4.apply(self.weights_init)
        self.out3.apply(self.weights_init)
        self.out2.apply(self.weights_init)

        self.shrink_pred1.apply(self.weights_init)
        self.shrink_pred2.apply(self.weights_init)
        self.origin_pred1.apply(self.weights_init)
        self.origin_pred2.apply(self.weights_init)
        self.dilate_pred.apply(self.weights_init)
        self.get_atten.apply(self.weights_init)
        self.up_trans.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def _init_pred1(self, inner_channels, bias):
        return nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True))

    def _init_pred2(self, inner_channels, bias):
        return nn.Sequential(
            nn.ConvTranspose2d(inner_channels // 4, inner_channels // 4, 2, 2),
            BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, 1, 2, 2),
            nn.Sigmoid())

    def forward(self, features, gt=None, masks=None, training=False):
        c2, c3, c4, c5 = features
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)

        in5_atten = self.get_atten(in5)
        self.normalize(in5_atten)
        in5_out = in5_atten * in5
        in6 = self.avgpool(in5_out)
        in6_1 = self.srm(in6)

        in6_2 = self.up_trans(in6_1)
        if in6_2.shape == in5.shape:
            in6_2 = in6_2 + in5
        else:
            bs, ch, h, w = in5.shape
            in6_2 = F.interpolate(in6_2, size=(h, w), mode='bilinear', align_corners=True) + in5

        out4 = self.up5(in6_2) + in4  # 1/16
        out3 = self.up4(out4) + in3  # 1/8
        out2 = self.up3(out3) + in2  # 1/4

        p5 = self.out5(in6_2)
        p4 = self.out4(out4)
        p3 = self.out3(out3)
        p2 = self.out2(out2)

        fuse = torch.cat((p5, p4, p3, p2), 1)
        shrink1 = self.shrink_pred1(fuse)
        origin1 = self.origin_pred1(fuse)

        fuse2 = torch.cat((shrink1, origin1), 1)
        dilate = self.dilate_pred(fuse2)
        shrink2 = self.shrink_pred2(shrink1)

        if self.training:
            origin2 = self.origin_pred2(origin1)
            result = OrderedDict(origin=origin2, shrink=shrink2, dilate=dilate)
        else:
            return shrink2, dilate
        return result

    def normalize(self, x):
        bs, ch, h, w = x.shape
        h = h - 1 if h % 2 == 1 else h
        w = w - 1 if w % 2 == 1 else w
        f1 = x[:, :, 0:h:2, 0:w:2]
        f2 = x[:, :, 0:h:2, 1:w:2]
        f3 = x[:, :, 1:h:2, 0:w:2]
        f4 = x[:, :, 1:h:2, 1:w:2]
        fuse = torch.cat((f1, f2, f3, f4), 1)
        fuse = F.softmax(fuse, dim=1)
        x[:, :, 0:h:2, 0:w:2] = fuse[:, 0, :, :].unsqueeze(1)
        x[:, :, 0:h:2, 1:w:2] = fuse[:, 1, :, :].unsqueeze(1)
        x[:, :, 1:h:2, 0:w:2] = fuse[:, 2, :, :].unsqueeze(1)
        x[:, :, 1:h:2, 1:w:2] = fuse[:, 3, :, :].unsqueeze(1)
