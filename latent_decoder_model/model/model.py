"""
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at the main github page.
Authors: Seung Wook Kim, Jonah Philion, Antonio Torralba, Sanja Fidler
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
try:
    from model.operations import *
except:
    from latent_decoder_model.model.operations import *

class styleVAEGAN(nn.Module):
    def __init__(self, size, n_mlp=8, channel_multiplier=2, args=None):
        super().__init__()
        self.z_dim = 64
        self.args = args
        lr_mlp = 0.01
        blur_kernel = [1,3,3,1]
        self.width_mul = 1
        if check_arg(args, 'width_mul'):
            self.width_mul = args.width_mul
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.theme_spat_dim = 32
        if check_arg(args, 'theme_spat_dim'):
            self.theme_spat_dim = args.theme_spat_dim
        self.spatial_dim = 4

        resb = ResBlock

        #######################
        ####### ENCODER #######
        #######################
        self.enc = nn.ModuleList()
        self.enc.append(ConvLayer(3, channels[size], 1))
        log_size = int(math.log(size, 2))
        in_channel = channels[size]

        # encoder residual blocks
        for i in range(log_size, int(math.log(self.spatial_dim, 2)), -1):
            out_channel = channels[2 ** (i - 1)]
            self.enc.append(resb(in_channel, out_channel))
            in_channel = out_channel

            if math.pow(2, i) == self.theme_spat_dim:
                # theme vector is going to be obtained from this layer
                self.theme_input_dim = out_channel

        # mu and logvar for spatial z
        self.to_vq_input = nn.Sequential(
            ConvLayer(channels[4], channels[4], 3),
            EqualConv2d(channels[4], 2*self.z_dim, 1, padding=0, stride=1)
        )
        # mu and logvar for theme z
        self.enc_theme_sampler = nn.Sequential(
            EqualConv2d(self.theme_input_dim, self.theme_input_dim, 1, padding=0, stride=1),
            nn.AvgPool2d((self.theme_spat_dim, int(self.width_mul*self.theme_spat_dim)), stride=(1,1)),
            View((-1, self.theme_input_dim)),
            EqualLinear(self.theme_input_dim,  self.args.theme_dim * 2)
        )


        #######################
        ####### DECODER #######
        #######################
        self.constant_input_size = self.spatial_dim
        self.style_dim = 512
        input_chan_dim = 512

        ####### spatial_z and theme_z processing layers
        self.spatial_z_process = nn.Sequential(
            EqualConv2d(self.z_dim, input_chan_dim, 1),
            nn.LeakyReLU(0.2)
        )
        self.spatial_z_merger = nn.Sequential(
            EqualConv2d(input_chan_dim * 2, channels[self.constant_input_size], 1),
            nn.LeakyReLU(0.2)
        )
        layers = [PixelNorm()]
        for i in range(n_mlp):
            in_dim, out_dim = self.style_dim, self.style_dim
            if i == 0:
                in_dim = self.args.theme_dim
            layers.append(
                EqualLinear(
                    in_dim, out_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )
        self.theme_z_process = nn.Sequential(*layers)

        ####### styleGAN generator architecture
        self.convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        self.constant_input = ConstantInput(input_chan_dim, size=self.constant_input_size, w=int(self.width_mul*self.constant_input_size))
        self.conv1 = StyledConv(
            channels[self.constant_input_size], channels[self.constant_input_size], 3, self.style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(channels[self.constant_input_size], self.style_dim, upsample=False)

        self.num_layers = (log_size - 2) * 2 + 1
        in_channel = channels[4]
        starting_i = int(math.log2(self.constant_input_size)) + 1
        for i in range(starting_i, log_size + 1):
            out_channel = channels[2 ** i]
            self.convs.append(StyledConv(in_channel, out_channel, 3, self.style_dim, upsample=True, blur_kernel=blur_kernel))
            self.convs.append(StyledConv(out_channel, out_channel, 3, self.style_dim, blur_kernel=blur_kernel))
            self.to_rgbs.append(ToRGB(out_channel, self.style_dim))
            in_channel = out_channel

        self.n_latent = log_size * 2 - 2

    def kl(self, mu, logvar):
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()
        return kl

    def reparameterize(self, param, mu_q=None, logvar_q=None, train=False):
        if mu_q is None:
            mu_q, logvar_q = torch.chunk(param, 2, dim=1)

        if not train:
            z = mu_q
        else:
            z = torch.randn_like(logvar_q) * torch.exp(0.5*logvar_q) + mu_q
        return z, mu_q, logvar_q

    def dec_preprocess(self, spatial_z, theme_z):
        spatial_latent = self.spatial_z_process(spatial_z)
        theme_latent = self.theme_z_process(theme_z)

        theme_latent = theme_latent.unsqueeze(1).repeat(1, self.n_latent, 1)
        static_input = self.constant_input(spatial_latent)
        spatial_latent = self.spatial_z_merger(torch.cat([static_input, spatial_latent], dim=1))
        return spatial_latent, theme_latent

    def get_value_from_dict(self, key, in_dict, rep_dict):
        if not (key in in_dict or key in rep_dict):
            return None
        if key in rep_dict:
            return rep_dict[key]
        return in_dict[key]

    def forward(self, input, replace_input={}, decode_only=False, return_latent_only=False, train=True):
        noise = [None] * self.num_layers  # no explicit noise for simplicity

        if decode_only:
            # decode from the given latent z
            theme_z = self.get_value_from_dict('theme_z', input, replace_input)
            spatial_z = self.get_value_from_dict('spatial_z', input, replace_input)
            spatial_latent, theme_latent = self.dec_preprocess(spatial_z, theme_z)

            out = self.conv1(spatial_latent, theme_latent[:, 0], noise=noise[0])
            skip = self.to_rgb1(out, theme_latent[:, 1])

            i = 1
            for conv1, conv2, noise1, noise2, to_rgb in zip(self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs):
                out = conv1(out, theme_latent[:, i], noise=noise1)
                out = conv2(out, theme_latent[:, i + 1], noise=noise2)
                skip = to_rgb(out, theme_latent[:, i + 2], skip)
                i += 2
            image = skip
            res = {'image': image, 'spatial_z': spatial_z, 'theme_z': theme_z,\
                    'theme_latent': theme_latent, 'spatial_latent': spatial_latent}
            return res

        ########################## encoding phase #########################
        enc_s, dists = {}, []
        s = input
        for enc_block in self.enc:
            enc_s[str(s.shape[2])] = s
            s = enc_block(s)
            if s.shape[2] == self.theme_spat_dim:
                theme_param = self.enc_theme_sampler(s)

        ## get spatial z
        gen_input = self.to_vq_input(s)
        spatial_z, spatial_mu, spatial_logvar = self.reparameterize(gen_input, train=train)

        ## get theme z
        theme_z, theme_mu, theme_logvar = self.reparameterize(theme_param, train=train)

        res = {}
        if return_latent_only:
            ## only encoding is needed, return all encoded zs
            theme_mu, theme_logvar = torch.chunk(theme_param, 2, dim=1)
            res['theme_mu'] = theme_mu
            res['theme_logvar'] = theme_logvar

            spatial_mu, spatial_logvar = torch.chunk(gen_input, 2, dim=1)
            res['spatial_mu'] = spatial_mu
            res['spatial_logvar'] = spatial_logvar
            return res

        ## define q for theme_z and spatial_z
        dists.append(['spatial', spatial_mu, spatial_logvar])
        dists.append(['theme', theme_mu, theme_logvar])


        ########################## decoding phase #########################

        spatial_latent, theme_latent = self.dec_preprocess(spatial_z, theme_z)
        out = self.conv1(spatial_latent, theme_latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, theme_latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs):
            out = conv1(out, theme_latent[:, i], noise=noise1)
            out = conv2(out, theme_latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, theme_latent[:, i + 2], skip)
            i += 2
        image = skip
        res = {'image': image, 'spatial_z': spatial_z, 'theme_z': theme_z, \
                'theme_latent': theme_latent, 'spatial_latent': spatial_latent}

        # compute kl
        kl_losses = {}
        for pq in dists:
            name, mu, logvar = pq
            kl_loss = self.kl(mu, logvar)
            kl_losses[name] = kl_loss
        res['kl_losses'] = kl_losses

        return res



class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, channel_multiplier2=1, blur_kernel=[1, 3, 3, 1], args=None):
        super().__init__()

        self.args = args
        channels = {
            4: int(512 * channel_multiplier2),
            8: int(512 * channel_multiplier2),
            16: int(512* channel_multiplier2),
            32: int(512* channel_multiplier2),
            64: int(256 * channel_multiplier* channel_multiplier2),
            128: int(128 * channel_multiplier* channel_multiplier2),
            256: int(64 * channel_multiplier* channel_multiplier2),
            512: int(32 * channel_multiplier* channel_multiplier2),
            1024: int(16 * channel_multiplier* channel_multiplier2),
        }

        self.theme_cut_layer = 4
        self.width_mul = 1
        if check_arg(args, 'width_mul'):
            self.width_mul = args.width_mul


        convs = [ConvLayer(3, channels[size], 1)]
        convs_top_spatial = []
        log_size = int(math.log(size, 2))
        added = 0
        in_channel = channels[size]
        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            added +=1
            convs.append(ResBlock(in_channel, out_channel, blur_kernel))
            in_channel = out_channel
        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * int(self.width_mul*4), channels[4], activation='fused_lrelu'),
            EqualLinear(channels[4], 1),
        )

        # multiscale patch discriminator
        self.Ds = nn.ModuleList()
        for i in range(self.args.num_patchD):
            self.Ds.append(SinglePatchDiscriminator(3))

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input, seg_input=None, seg_index=-1):
        results = []

        out_feature_top = self.convs(input)
        batch, channel, height, width = out_feature_top.shape
        group = batch
        stddev = out_feature_top.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out_feature_top, stddev], 1)

        out = self.final_conv(out)
        out = out.view(batch, -1)
        out = self.final_linear(out)
        results.append(out)
        for i in range(len(self.Ds)):
            result = self.Ds[i](input)
            results.append(result)
            input = self.downsample(input)

        return results


class SinglePatchDiscriminator(nn.Module):
    def __init__(self, in_channel, blur_kernel=[1, 3, 3, 1], n_downsample=4):
        super().__init__()

        channels = [128, 256, 512, 512, 512]

        convs = [ConvLayer(in_channel, channels[0], 1)]
        in_channel = channels[0]
        self.n_downsample = n_downsample
        for i in range(n_downsample):
            out_channel = channels[i+1]
            convs.append(ResBlock(in_channel, out_channel, blur_kernel))
            in_channel = out_channel

        convs.append(EqualConv2d(in_channel, 1, 3, stride=1, padding=1))
        self.convs = nn.Sequential(*convs)

    def forward(self, input):
        bs = input.size(0)
        out = self.convs(input)
        out = out.view(bs, -1)
        return out
