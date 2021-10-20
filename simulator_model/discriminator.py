"""
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at the main github page.
Authors: Seung Wook Kim, Jonah Philion, Antonio Torralba, Sanja Fidler
"""

import torch
from torch import nn
import torch.utils.data
import torch.utils.data.distributed
import numpy as np
import utils
import sys
sys.path.append('..')
import torch.nn.utils.spectral_norm as SN

from simulator_model.model_utils import View
from simulator_model import model_utils
from simulator_model import layers
import functools
from torch.nn import init
import random


class DiscriminatorSingleLatent(nn.Module):
    def __init__(self, opts):
        super(DiscriminatorSingleLatent, self).__init__()
        self.opts = opts
        dim = opts.nfilterD * 16

        self.which_linear = functools.partial(layers.SNLinear,
                                              num_svs=1, num_itrs=1,
                                              eps=1e-12)

        sinput_dim = opts.latent_z_size
        l = [self.which_linear(sinput_dim, dim)]
        l.append(nn.BatchNorm1d(dim)),
        l.append(nn.LeakyReLU(0.2))

        num_layers = 3
        if utils.check_arg(opts, 'D_num_base_layer'):
            num_layers = opts.D_num_base_layer

        for _ in range(num_layers):
            l.append(self.which_linear(dim, dim))
            l.append(nn.BatchNorm1d(dim))
            l.append(nn.LeakyReLU(0.2))
        self.base = nn.Sequential(*l)

        self.d_final = nn.Sequential(self.which_linear(dim, dim),
                                     nn.BatchNorm1d(dim),
                                     nn.LeakyReLU(0.2),
                                     self.which_linear(dim, 1))


    def forward(self, x):

        h = self.base(x)
        return self.d_final(h), h


class Discriminator(nn.Module):

    def __init__(self, opts, nfilter=32, nfilter_max=1024):
        super(Discriminator, self).__init__()

        self.opts = opts
        self.disentangle_style = utils.check_arg(self.opts, 'disentangle_style')
        self.separate_holistic_style_dim = self.opts.separate_holistic_style_dim
        f_size = 4

        self.ds = DiscriminatorSingleLatent(opts)
        conv3d_dim = opts.nfilterD_temp * 16

        self.temporal_window = self.opts.config_temporal

        self.conv3d, self.conv3d_final = \
            model_utils.choose_netD_temporal(
                self.opts, conv3d_dim, window=self.temporal_window
            )
        self.conv3d = nn.ModuleList(self.conv3d)
        self.conv3d_final = nn.ModuleList(self.conv3d_final)


        self.which_conv = functools.partial(layers.SNConv2d,
                                            kernel_size=f_size, padding=0,
                                            num_svs=1, num_itrs=1,
                                            eps=1e-12)
        self.which_linear = functools.partial(layers.SNLinear,
                                              num_svs=1, num_itrs=1,
                                              eps=1e-12)

        # For action discriminator
        self.trans_conv = self.which_linear(opts.nfilterD*16*2, opts.nfilterD*16)
        self.to_transition_feature = nn.Sequential(self.trans_conv,
                                                   nn.LeakyReLU(0.2),
                                                   View((-1, opts.nfilterD*16)))

        action_space = 10 if not utils.check_arg(self.opts, 'action_space') else self.opts.action_space
        self.action_to_feat = nn.Linear(action_space, opts.nfilterD*16)


        self.reconstruct_action_z = self.which_linear(opts.nfilterD*16, action_space)  # 4, 1, 0),




    def forward(self, images, actions, states, warm_up, neg_actions=None, epoch=0):
        dout = {}
        neg_action_predictions, rev_predictions, content_predictions = None, None, []
        neg_content_predictions, action_predictions = None, None
        batch_size = actions[0].size(0)


        if warm_up == 0:
            warm_up = 1 # even if warm_up is 0, the first screen is from GT

        gt_states = torch.cat(states[:warm_up], dim=0)

        single_frame_predictions_all, tmp_features = self.ds(torch.cat([gt_states, images], dim=0))
        single_frame_predictions_all = single_frame_predictions_all[warm_up * batch_size:]
        frame_features = tmp_features[warm_up*batch_size:]
        next_features = frame_features
        # action discriminator
        prev_frames = torch.cat([tmp_features[:warm_up*batch_size],
                                 tmp_features[(warm_up+warm_up-1)*batch_size:-batch_size]], dim=0)

        if self.opts.input_detach:
            prev_frames = prev_frames.detach()
        transition_features = self.to_transition_feature(torch.cat([prev_frames, next_features], dim=1))
        action_features = self.action_to_feat(torch.cat(actions[:-1], dim=0))
        if neg_actions is not None:
            neg_action_features = self.action_to_feat(torch.cat(neg_actions[:-1], dim=0))

        action_recon = self.reconstruct_action_z(transition_features)

        new_l = []
        temporal_predictions = []


        stacked = torch.cat([action_features, transition_features], dim=1)
        stacked = stacked.view(len(actions)-1, batch_size, -1).permute(1,0,2)
        stacked = stacked.permute(0, 2, 1)

        if neg_actions is not None:
            neg_stacked = torch.cat([neg_action_features, transition_features], dim=1)
            neg_stacked = neg_stacked.view(len(actions) - 1, batch_size, -1).permute(1, 0, 2)
            neg_stacked = neg_stacked.permute(0, 2, 1)
            if self.opts.do_latent:
                neg_stacked = neg_stacked.unsqueeze(-1)

            neg_content_predictions = []
            aa = self.conv3d[0](neg_stacked)
            a_out = self.conv3d_final[0](aa)
            neg_content_predictions.append(a_out.view(batch_size, -1))
            if self.temporal_window >= 12:
                bb = self.conv3d[1](aa)
                b_out = self.conv3d_final[1](bb)
                neg_content_predictions.append(b_out.view(batch_size, -1))
            if self.temporal_window >= 18:
                cc = self.conv3d[2](bb)
                c_out = self.conv3d_final[2](cc)
                neg_content_predictions.append(c_out.view(batch_size, -1))
            if self.temporal_window >= 30:
                dd = self.conv3d[3](cc)
                d_out = self.conv3d_final[3](dd)
                neg_content_predictions.append(d_out.view(batch_size, -1))


        stacked = stacked.unsqueeze(-1)

        aa = self.conv3d[0](stacked)
        a_out = self.conv3d_final[0](aa)
        temporal_predictions.append(a_out.view(batch_size, -1))
        if self.temporal_window >= 12:
            bb = self.conv3d[1](aa)
            b_out = self.conv3d_final[1](bb)
            temporal_predictions.append(b_out.view(batch_size, -1))
        if self.temporal_window >= 18:
            cc = self.conv3d[2](bb)
            c_out = self.conv3d_final[2](cc)
            temporal_predictions.append(c_out.view(batch_size, -1))
        if self.temporal_window >= 36:
            dd = self.conv3d[3](cc)
            d_out = self.conv3d_final[3](dd)
            temporal_predictions.append(d_out.view(batch_size, -1))

        dout['disc_features'] = frame_features[:(len(states)-1)*batch_size]
        dout['action_predictions'] = action_predictions
        dout['single_frame_predictions_all'] = single_frame_predictions_all
        dout['content_predictions'] = temporal_predictions
        dout['neg_action_predictions'] = neg_action_predictions
        dout['neg_content_predictions'] = neg_content_predictions
        dout['action_recon'] = action_recon
        return dout

    def update_opts(self, opts):
        self.opts = opts
        return
