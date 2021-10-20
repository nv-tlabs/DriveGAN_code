"""
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at the main github page.
Authors: Seung Wook Kim, Jonah Philion, Antonio Torralba, Sanja Fidler
"""

"""
Contains some code from:
https://github.com/LMescheder/GAN_stability
with the following license:
MIT License

Copyright (c) 2018 Lars Mescheder

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import utils
import torch
import torch.nn.functional as F
import torch.utils.data
import math
import numpy as np

class Trainer(object):
    def __init__(self, opts,
                 netG, netD,
                 optG_temporal, optG_graphic, optD,
                 reg_param):

        self.opts = opts

        self.netG = netG
        self.netG.opts = opts
        self.netD = netD
        if self.netD is not None:
            self.netD.opts = opts

        self.optG_temporal = optG_temporal
        self.optG_graphic = optG_graphic
        self.optD = optD

        self.reg_param = reg_param

        # Default to hinge loss
        if utils.check_arg(opts, 'standard_gan_loss'):
            self.generator_loss = self.standard_gan_loss
            self.discriminator_loss = self.standard_gan_loss
        else:
            self.generator_loss = self.loss_hinge_gen
            self.discriminator_loss = self.loss_hinge_dis

    # Hinge loss for discriminator
    def loss_hinge_dis(self, logits, label, masking=None, div=None):
        if label == 1:
            t = F.relu(1. - logits)
        else:
            t = F.relu(1. + logits)

        if div is None:
            return torch.mean(t)
        else:
            assert(masking is not None)
            t = t * masking
            return torch.sum(t) / div

    # Hinge loss for generator
    def loss_hinge_gen(self, dis_fake):
        loss = -torch.mean(dis_fake)
        return loss

    # BCE GAN loss
    def standard_gan_loss(self, d_out, target=1):
        if d_out is None:
            return utils.check_gpu(self.opts.gpu, torch.FloatTensor([0]))
        targets = d_out.new_full(size=d_out.size(), fill_value=target)

        loss = F.binary_cross_entropy_with_logits(d_out, targets)
        return loss

    # Reconstruction loss
    def get_recon_loss(self, input, target, detach=True, criterion=None, div=None):

        if div is None:
            div = target.size(0)
        if detach:
            target = target.detach()
        loss = criterion(input, target, size_average=False) / div
        return loss

    def merge_dicts(self, d1, d2, d2name=''):
        for key, val in d2.items():
            d1[d2name+'_'+key] = val
        return d1

    def get_num_repeat(self, input_like, bs):
        return input_like.size(0) // bs

    def calculate_discriminator_adv_loss_fake(self, loss_dict, gout, dout_fake):
        ## temporal loss
        dloss_fake_content_loss = 0
        if self.opts.temporal_hierarchy:
            for i in range(len(dout_fake['content_predictions'])):
                curloss = self.discriminator_loss(dout_fake['content_predictions'][i], 0)
                loss_dict['dloss_fake_content_loss' + str(i)] = curloss
                dloss_fake_content_loss += curloss
            dloss_fake_content_loss = dloss_fake_content_loss / len(dout_fake['content_predictions'])
            loss_dict['dloss_fake_content_loss'] = dloss_fake_content_loss

        ## single frame loss
        dloss_fake_single_frame_loss = self.discriminator_loss(dout_fake['single_frame_predictions_all'], 0)
        loss_dict['dloss_fake_single_frame_loss'] = dloss_fake_single_frame_loss

        loss = dloss_fake_content_loss + dloss_fake_single_frame_loss


        return loss_dict, loss

    def calculate_generator_adv_loss(self, loss_dict, dout_fake, dout_real, gen_actions, gout, bs):
        ## temporal loss
        gloss_content_loss = 0
        if self.opts.temporal_hierarchy:
            for i in range(len(dout_fake['content_predictions'])):
                curloss = self.generator_loss(dout_fake['content_predictions'][i])
                loss_dict['gloss_content_loss' + str(i)] = curloss
                gloss_content_loss += curloss

        ## single frame loss
        gloss_single_frame_loss = self.generator_loss(dout_fake['single_frame_predictions_all'])
        loss_dict['gloss_single_frame_loss'] = gloss_single_frame_loss

        ## loss to recover action from frames
        action_orig = torch.cat(gen_actions[:len(gout['outputs'])], dim=0)
        action_recon_loss = F.mse_loss(dout_fake['action_recon'], action_orig)
        loss_dict['g_action_recon_loss'] = action_recon_loss

        total_loss = self.opts.gen_content_loss_multiplier * gloss_content_loss + \
                      gloss_single_frame_loss + action_recon_loss

        return loss_dict, total_loss


    def get_kl_loss(self, gout, name, beta, loss_dict):
        kl_loss = gout[name].mean()
        loss_dict[name] = kl_loss
        return kl_loss * beta


    def generator_trainstep(self, states, actions, warm_up=10, train=True, epoch=0, it=0, latent_decoder=None):
        '''
        Run one step of generator
        '''
        bs = states[0].size(0)

        # set number of warm up images
        if self.opts.warmup_decay_step > 0:
            warm_up = max(self.opts.min_warmup, math.ceil(warm_up * (1 - it * 1.0 / self.opts.warmup_decay_step)))

        utils.toggle_grad(self.netG, True)
        utils.toggle_grad(self.netD, True)
        if train:
            self.netG.train()
            self.netD.train()
        else:
            self.netG.eval()
            self.netD.eval()

        self.optD.zero_grad()
        self.optG_temporal.zero_grad()
        self.optG_graphic.zero_grad()

        loss_dict = {}
        gen_actions = actions

        # generate the output sequence
        gout = self.netG(states, gen_actions, warm_up, train=train, epoch=epoch, latent_decoder=latent_decoder)

        graphic_loss, temporal_loss, total_loss, dout_fake, rev_images = 0, 0, 0, None, None
        if self.opts.gan_loss:
            ######################### fool discriminator ########################
            # generated sequence
            gen_adv_input = torch.cat(gout['outputs'], dim=0)
            dout_fake = self.netD(gen_adv_input, gen_actions[:len(gout['outputs']) + 1], states, warm_up)

            # real sequence
            din = states[1:len(gout['outputs']) + 1]
            dout_real = self.netD(torch.cat(din, dim=0), actions[:len(gout['outputs']) + 1], states, warm_up)

            loss_dict, total_loss = self.calculate_generator_adv_loss({}, dout_fake, dout_real, gen_actions, gout, bs)

            if self.opts.disc_features:
                ## feature matching loss
                feat_loss_fn = F.l1_loss
                x_fake_ = dout_fake['disc_features']
                x_real_ = dout_real['disc_features'].detach()
                loss_l1_disc_features = feat_loss_fn(x_fake_, x_real_, reduction='none').mean(1).sum(0) / x_fake_.size(0)
                loss_dict['loss_l1_disc_features'] = loss_l1_disc_features
                total_loss += self.opts.feature_loss_multiplier * (loss_l1_disc_features)

        ## recon_loss
        recon_multiplier = self.opts.recon_loss_multiplier
        x_fake_ = torch.cat(gout['outputs'], dim=0)
        x_real_ = torch.cat(states[1:len(gout['outputs']) + 1], dim=0)
        criterion = F.mse_loss

        loss_recon_spatial = self.get_recon_loss(x_fake_[:, :self.opts.spatial_total_dim], x_real_[:, :self.opts.spatial_total_dim], criterion=criterion, div=x_fake_.size(0))
        loss_recon_theme = self.get_recon_loss(x_fake_[:, self.opts.spatial_total_dim:], x_real_[:, self.opts.spatial_total_dim:], criterion=criterion, div=x_fake_.size(0))
        loss_dict['loss_recon_spatial'] = loss_recon_spatial
        loss_dict['loss_recon_theme'] = loss_recon_theme
        total_loss += recon_multiplier * (loss_recon_spatial + loss_recon_theme)

        ## kl losses
        if self.opts.disentangle_style:
            total_loss += self.get_kl_loss(gout, 'z_aindep_kl_loss', self.opts.style_kl_beta, loss_dict)
        if self.opts.theme_d > 0:
            total_loss += self.get_kl_loss(gout, 'z_theme_kl_loss', self.opts.theme_kl_beta, loss_dict)

        z_adep_kl_loss = gout['z_adep_kl_loss'].mean()
        if z_adep_kl_loss > 0.1: # thresholding
            total_loss += self.opts.content_kl_beta * z_adep_kl_loss
        loss_dict['z_adep_kl_loss'] = z_adep_kl_loss

        if train:
            total_loss.backward()
            self.optG_temporal.step()
            self.optG_graphic.step()

        return loss_dict, total_loss, gout

    def discriminator_trainstep(self, states, actions, neg_actions, warm_up=10, gout=None,
                                epoch=0, step=0, it=0):
        '''
        Run one step of discriminator
        '''
        if self.opts.warmup_decay_step > 0:
            warm_up = max(self.opts.min_warmup, math.ceil(warm_up * (1 - it * 1.0 / self.opts.warmup_decay_step)))

        utils.toggle_grad(self.netG, False)
        utils.toggle_grad(self.netD, True)
        self.netG.train()
        self.netD.train()

        self.optG_temporal.zero_grad()
        self.optG_graphic.zero_grad()

        self.optD.zero_grad()

        loss_dict = {}
        states = [x.requires_grad_() for x in states]
        actions = [x.requires_grad_() for x in actions]
        neg_actions = [x.requires_grad_() for x in neg_actions]

        ################# On real data ####################
        d_input = torch.cat(states[1:], dim=0)
        d_input = d_input.requires_grad_()
        dout = self.netD(d_input, actions, states, warm_up, neg_actions=neg_actions)

        loss = 0
        dloss_real_action, dloss_real_action_wrong = 0, 0
        dloss_real_content_action_wrong_loss = 0

        # single frame loss
        dloss_real_single_frame_loss = self.discriminator_loss(dout['single_frame_predictions_all'], 1)
        loss_dict['dloss_real_single_frame_loss'] = dloss_real_single_frame_loss

        # temporal loss - false actions
        if self.opts.temporal_hierarchy:
            for i in range(len(dout['neg_content_predictions'])):
                curloss = self.discriminator_loss(dout['neg_content_predictions'][i], 0)
                loss_dict['dloss_real_content_action_wrong_loss' + str(i)] = curloss
                dloss_real_content_action_wrong_loss += curloss
            dloss_real_content_action_wrong_loss = dloss_real_content_action_wrong_loss / len(dout['content_predictions'])
            loss_dict['dloss_real_content_action_wrong_loss'] = dloss_real_content_action_wrong_loss

        # action reconstruction loss
        action_orig = torch.cat(actions[:-1], dim=0)
        action_recon_loss = F.mse_loss(dout['action_recon'], action_orig)
        loss_dict['d_action_recon_loss'] = action_recon_loss

        # temporal loss - true actions
        dloss_real_content_loss = 0
        if self.opts.temporal_hierarchy:
            for i in range(len(dout['content_predictions'])):
                curloss = self.discriminator_loss(dout['content_predictions'][i], 1)
                loss_dict['dloss_real_content_loss' + str(i)] = curloss
                dloss_real_content_loss += curloss
            dloss_real_content_loss = dloss_real_content_loss / len(dout['content_predictions'])
            loss_dict['dloss_real_content_loss'] = dloss_real_content_loss


        loss += (dloss_real_content_loss + 0.2* dloss_real_content_action_wrong_loss) + \
            dloss_real_single_frame_loss + dloss_real_action + dloss_real_action_wrong + action_recon_loss

        # regularization
        reg = 0
        if self.reg_param > 0:
            reg += 0.33*utils.compute_grad2(dout['single_frame_predictions_all'], d_input, ns=self.opts.num_steps).mean()
            reg += 0.33*utils.compute_grad2(dout['action_recon'], d_input, ns=self.opts.num_steps).mean()

            reg_temporal = 0
            if self.opts.temporal_hierarchy:
                for i in range(len(dout['content_predictions'])):
                    curloss = utils.compute_grad2(dout['content_predictions'][i], d_input, ns=self.opts.num_steps).mean()
                    reg_temporal += curloss
                reg_temporal = reg_temporal / len(dout['content_predictions'])
                loss_dict['dloss_REG_temporal'] = reg_temporal

            loss_dict['dloss_REG'] = reg
            loss += self.reg_param * reg + self.opts.LAMBDA_temporal * reg_temporal


        ################# On fake data ####################
        gen_actions = actions
        dout_fake = self.netD(torch.cat(gout['outputs'], dim=0).detach(), gen_actions[:len(gout['outputs']) + 1],
                              states, warm_up)
        loss_dict, fake_loss = self.calculate_discriminator_adv_loss_fake(loss_dict, gout, dout_fake)
        loss += fake_loss

        loss.backward()
        self.optD.step()
        utils.toggle_grad(self.netD, False)

        return loss_dict
