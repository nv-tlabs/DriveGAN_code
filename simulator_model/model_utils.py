"""
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at the main github page.
Authors: Seung Wook Kim, Jonah Philion, Antonio Torralba, Sanja Fidler
"""
import torch
from torch import nn
from torch.nn import functional as F
from simulator_model import layers
import functools
import sys
sys.path.append('..')

class convLinearSPADE(nn.Module):
    def __init__(self, channel, h, w, linear_input_channel, opts):
        super().__init__()
        self.h = h
        self.w = w
        self.param_free_norm = nn.InstanceNorm2d(channel, affine=False)

        self.mlp_gamma = nn.Linear(linear_input_channel, channel)
        self.mlp_beta = nn.Linear(linear_input_channel, channel)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, y, resize=True):
        if resize:
            x = x.view(x.size(0), -1, self.h, self.w)
        normalized = self.param_free_norm(x)
        y = y.view(y.size(0), -1)
        gamma = self.mlp_gamma(y).view(y.size(0), -1, 1, 1)
        beta = self.mlp_beta(y).view(y.size(0), -1, 1, 1)

        out = normalized * (1 + gamma) + beta

        return self.activation(out)

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


def choose_netG_encoder(input_dim=512, basechannel=512, opts=None):
    enc = nn.Sequential(
        nn.Linear(input_dim, basechannel),
        nn.LeakyReLU(0.2),
        nn.Linear(basechannel, basechannel),
        nn.LeakyReLU(0.2),
        nn.Linear(basechannel, basechannel),
        nn.LeakyReLU(0.2),
        nn.Linear(basechannel, basechannel),
        nn.LeakyReLU(0.2)
    )

    return enc



def choose_netD_temporal(opts, conv3d_dim, window=[]):
    in_dim = opts.nfilterD * 16
    in_dim = in_dim * 2
    extractors, finals = [], []

    which_conv = functools.partial(layers.SNConv2d,
                                        kernel_size=3, padding=0,
                                        num_svs=1, num_itrs=1,
                                        eps=1e-12)

    net1 = nn.Sequential(
        which_conv(in_dim, conv3d_dim // 4, kernel_size=(3, 1), stride=(2, 1)),
        nn.LeakyReLU(0.2)
    )

    head1 = nn.Sequential(
        which_conv(conv3d_dim // 4, 1, kernel_size=(2, 1), stride=(1, 1)),
    )
    extractors.append(net1)
    finals.append(head1)

    if window >= 12:
        net2 = nn.Sequential(
            which_conv(conv3d_dim // 4, conv3d_dim // 2, kernel_size=(3, 1), stride=(1, 1)),
            nn.LeakyReLU(0.2),
        )
        head2 = nn.Sequential(
            which_conv(conv3d_dim // 2, 1, kernel_size=(3, 1)),
        )
        extractors.append(net2)
        finals.append(head2)

    if window >= 18:
        net3 = nn.Sequential(
            which_conv(conv3d_dim // 2, conv3d_dim, kernel_size=(2, 1), stride=(2, 1)),
            nn.LeakyReLU(0.2),
        )
        head3 = nn.Sequential(
            which_conv(conv3d_dim, 1, kernel_size=(3, 1)),
        )
        extractors.append(net3)
        finals.append(head3)

    if window >= 36:
        net4 = nn.Sequential(
            which_conv(conv3d_dim, conv3d_dim, kernel_size=(2, 1), stride=(2, 1)),
            nn.LeakyReLU(0.2),
        )
        head4 = nn.Sequential(
            which_conv(conv3d_dim, 1, kernel_size=(3, 1)),
        )
        extractors.append(net4)
        finals.append(head4)

    return extractors, finals
