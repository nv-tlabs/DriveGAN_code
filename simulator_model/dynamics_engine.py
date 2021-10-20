"""
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at the main github page.
Authors: Seung Wook Kim, Jonah Philion, Antonio Torralba, Sanja Fidler
"""

import torch
from torch import nn
import utils
import sys
sys.path.append('..')
import numpy as np
from simulator_model import model_utils
import torch.nn.functional as F


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size=1024, opts=None):
        super(ConvLSTM, self).__init__()
        self.opts = opts
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.conv_lstm_num_layer = 2
        if utils.check_arg(opts, 'conv_lstm_num_layer'):
            self.conv_lstm_num_layer = opts.conv_lstm_num_layer

        filter_size, padding = 3, 1
        if self.conv_lstm_num_layer == 1 and self.opts.spatial_h == 4 and self.opts.spatial_w == 6:
            filter_size, padding = (3,5), (1,2)
            conv = [nn.Conv2d(self.hidden_size +self.input_dim, 4 * self.hidden_size, filter_size, stride=1, padding=padding)]

        else:
            if self.conv_lstm_num_layer == 2 and self.opts.spatial_h == 4 and self.opts.spatial_w == 6:
                filter_size, padding = (3,5), (1,2)

            conv = [nn.Conv2d(self.hidden_size +self.input_dim, self.hidden_size, filter_size, stride=1, padding=padding),
                     nn.LeakyReLU(0.2)]

            for i in range(self.conv_lstm_num_layer-2):
                conv.append( nn.Conv2d(self.hidden_size, self.hidden_size, filter_size, stride=1, padding=padding))
                conv.append( nn.LeakyReLU(0.2))

            conv.append(nn.Conv2d(self.hidden_size, 4 * self.hidden_size, filter_size, stride=1, padding=padding))
        self.conv = nn.Sequential(*conv)
    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size, self.opts.spatial_h, self.opts.spatial_w), \
               torch.zeros(bs, self.hidden_size, self.opts.spatial_h, self.opts.spatial_w)

    def forward(self, h, c, input):
        v = torch.cat([h, input], dim=1)
        tmp = self.conv(v)
        g_t = tmp[:, 3 * self.hidden_size:].tanh()
        i_t = tmp[:, :self.hidden_size].sigmoid()
        f_t = tmp[:, self.hidden_size:2 * self.hidden_size].sigmoid()
        o_t = tmp[:, 2 * self.hidden_size:3 * self.hidden_size].sigmoid()
        c_t = torch.mul(c, f_t) + torch.mul(i_t, g_t)
        h_t = torch.mul(o_t, torch.tanh(c_t))

        return h_t, c_t


class EngineModule(nn.Module):
    def __init__(self, opts, state_dim):
        super(EngineModule, self).__init__()
        self.hdim = opts.hidden_dim
        self.opts = opts
        action_space = 10 if not utils.check_arg(self.opts, 'action_space') else self.opts.action_space

        # engine module input network
        e_input_dim = self.hdim
        sinput_dim = (opts.latent_z_size - self.opts.theme_d) // (self.opts.spatial_h * self.opts.spatial_w)
        lstm_input_dim = opts.convLSTM_hidden_dim // 8
        sinput_dim = int(sinput_dim)
        # action embedding
        self.a_to_input = nn.Sequential(nn.Linear(action_space, lstm_input_dim),
                                        nn.LeakyReLU(0.2))
        # layer for processing spatial z
        self.s_to_input = nn.Sequential(
            nn.Conv2d(sinput_dim, lstm_input_dim, 1),
            nn.LeakyReLU(0.2))

        e_input_dim = lstm_input_dim * 2
        if self.opts.theme_d > 0:
            # layer for processing theme z
            self.holistic_to_input = nn.Sequential(nn.Linear(self.opts.theme_d, lstm_input_dim),
                                        nn.LeakyReLU(0.2))
            e_input_dim += lstm_input_dim

        # the main conv-lstm module
        self.rnn_e = ConvLSTM(e_input_dim,
                                hidden_size=opts.convLSTM_hidden_dim, opts=opts)


    def init_hidden(self, bs):
        h, c = self.rnn_e.init_hidden(bs)
        return [h], [c]

    def forward(self, h, c, s, a):
        '''
        h, c: hidden and cell state of lstm
        s: concatenated input spatial and theme z
        a: input action
        '''
        h_e = h[0]
        c_e = c[0]
        input_core = [self.a_to_input(a)]

        # Core engine
        h_e = h_e.view(h_e.size(0), -1, self.opts.spatial_h, self.opts.spatial_w)
        c_e = c_e.view(h_e.size(0), -1, self.opts.spatial_h, self.opts.spatial_w)

        if self.opts.theme_d > 0:
            content = s[:, :-self.opts.theme_d]
            theme = s[:, -self.opts.theme_d:]
            theme_input = self.holistic_to_input(theme)
            input_core.append(theme_input)
        else:
            content = s
        content = self.s_to_input(content.view(content.size(0), -1, self.opts.spatial_h, self.opts.spatial_w))
        input_core = torch.cat(input_core, dim=1).unsqueeze(2).unsqueeze(3)
        input_core = input_core.repeat(1, 1, self.opts.spatial_h, self.opts.spatial_w)
        input_core = torch.cat([input_core, content], dim=1)

        h_e_t, c_e_t = self.rnn_e(h_e, c_e, input_core)
        h_e_t = h_e_t.view(h_e.size(0), -1)
        c_e_t = c_e_t.view(h_e.size(0), -1)

        H = [h_e_t]
        C = [c_e_t]

        return H, C, h_e_t

class EngineGenerator(nn.Module):

    def __init__(self, opts, nfilter_max=512, **kwargs):
        super(EngineGenerator, self).__init__()

        self.opts = opts
        self.num_content_kl_added = 0
        self.num_z_aindep_kl_loss_added = 0
        self.num_theme_kl_added = 0

        self.hdim = opts.hidden_dim
        self.base_dim = opts.hidden_dim
        self.disentangle_style = utils.check_arg(self.opts, 'disentangle_style')

        if utils.check_arg(self.opts, 'theme_d'):
            self.theme_d = self.opts.theme_d
        else:
            self.theme_d = 0

        state_dim = opts.latent_z_size
        self.state_dim = state_dim

        # Dynamics Engine, ConvLSTM
        self.engine = EngineModule(opts, state_dim)

        # action-dependent content head that processes the output of convLSTM
        self.content_enc_head_dim = int(opts.hidden_dim // (self.opts.spatial_h*self.opts.spatial_w))
        self.content_enc_head = nn.Conv2d(opts.convLSTM_hidden_dim, self.content_enc_head_dim * 2, 1)

        # lstm and related modules for processing action-independent content
        self.style_enc = model_utils.choose_netG_encoder(input_dim=opts.latent_z_size, basechannel=opts.hidden_dim, opts=self.opts)
        self.style_enc_head = None
        self.style_lstm = nn.GRU(opts.hidden_dim, opts.hidden_dim, 1, batch_first=True)
        self.residual_enc_head = nn.Linear(opts.hidden_dim, opts.hidden_dim * 2)

        if self.theme_d > 0:
            ## mlp for theme vector, that processes the output of convLSTM
            theme_enc_head_dim = self.theme_d * 2
            self.holistic_style_enc_head = nn.Linear(opts.convLSTM_hidden_dim * (self.opts.spatial_h*self.opts.spatial_w), theme_enc_head_dim)
            self.holistic_style_mlp = nn.Sequential(
                nn.Linear(self.theme_d, self.theme_d),
                nn.LeakyReLU(0.2),
                nn.Linear(self.theme_d, self.theme_d),
            )

        # layers for combining action-dependent and action-independent content zs
        self.style_mlp1 = nn.Sequential(
            nn.Linear(opts.hidden_dim, opts.hidden_dim),
            nn.LeakyReLU(0.2),
            model_utils.View((-1, opts.hidden_dim // (self.opts.spatial_h*self.opts.spatial_w), self.opts.spatial_h, self.opts.spatial_w))
        )
        self.modulate_style1 = model_utils.convLinearSPADE(int(opts.hidden_dim // (self.opts.spatial_h*self.opts.spatial_w)), self.opts.spatial_h, self.opts.spatial_w, opts.hidden_dim, opts=opts)
        self.style_mlp2 = nn.Sequential(
            nn.Conv2d(opts.hidden_dim // (self.opts.spatial_h*self.opts.spatial_w), opts.hidden_dim // 4, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.modulate_style2 = model_utils.convLinearSPADE(int(opts.hidden_dim // 4), self.opts.spatial_h, self.opts.spatial_w, opts.hidden_dim, opts=opts)

        # final mlp for producing the output content z
        sinput_dim = (opts.latent_z_size - self.opts.theme_d) // (self.opts.spatial_h*self.opts.spatial_w)
        renderer_layers = [nn.Conv2d(opts.hidden_dim // 4, sinput_dim, 3, 1, 1)]
        self.graphics_renderer = nn.Sequential(*renderer_layers)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def kl(self, logvar, mu):
        kl_loss = -0.5 * torch.mean(-logvar.exp() - torch.pow(mu, 2) + logvar + 1, 1)
        return kl_loss

    def run_style_lstm(self, state, force_style, style_h, modules, step, \
                       play, reset=False, logvars_style=None):
        '''
        run one step of linear action-independent lstm
        '''
        style_enc, style_enc_head, residual_enc_head, style_lstm = modules
        if reset:
            style_h = None

        if force_style is not None:
            ## force input
            style_lstm_input = force_style
        else:
            style_lstm_input = style_enc(state)

        residual, style_h = style_lstm(style_lstm_input.unsqueeze(1), style_h)
        residual = residual_enc_head(residual.squeeze(1))
        style_mu, style_logvar = residual.split(self.opts.hidden_dim, dim=1)
        if logvars_style is not None and len(logvars_style) > step:
            # force logvar to be predefined for testing purposes
            eps = F.tanh(logvars_style[step])
            std = torch.exp(0.5 * style_logvar)
            style_reparam_vec = style_mu + std * eps
        else:
            style_reparam_vec = self.reparam(style_mu, style_logvar) if not play else style_mu

        z_aindep_kl_loss = 0
        if not play:
            z_aindep_kl_loss = self.kl(style_logvar, style_mu)
        self.num_z_aindep_kl_loss_added += 1
        z_aindep = style_reparam_vec.view(style_reparam_vec.size(0), -1)

        return z_aindep, style_h, z_aindep_kl_loss

    def run_vae_hidden(self, content_out, enc_head, size, play, kind='content', reshape=True, step=0, logvars=None):
        if reshape:
            content_out = content_out.view(content_out.size(0), -1, self.opts.spatial_h, self.opts.spatial_w)

        mu, logvar = enc_head(content_out).split(size, dim=1)
        if logvars is not None and len(logvars) > step:
            # force logvar to be predefined for testing purposes
            eps = logvars[step]
            std = torch.exp(0.5 * logvar)
            vec = mu + std * eps
        else:
            vec = self.reparam(mu, logvar) if not play else mu

        kl_loss = 0
        if not play:
            kl_loss = self.kl(logvar, mu)

        if kind == 'content':
            self.num_content_kl_added += 1
        elif kind == 'theme':
            self.num_theme_kl_added += 1

        vec = vec.view(vec.size(0), -1)

        return vec, kl_loss

    def run_input_enc(self, force_init_state, batch_size, state, h, c):
        if force_init_state is not None:
            content_enc_input = force_init_state
            h, c = self.engine.init_hidden(batch_size)
            h = utils.check_gpu(self.opts.gpu, h)
            c = utils.check_gpu(self.opts.gpu, c)
        else:
            content_enc_input = state

        return content_enc_input, h, c

    def merge_z_content(self, z_aindep, z_adep):
        '''
        Run AdaIN layers to combine action-dependent with action-independent z
        and final mlp layers
        '''
        style_modulate_vec = self.style_mlp1(z_aindep)
        input_to_renderer = self.modulate_style1(z_adep, style_modulate_vec)
        input_to_renderer = self.style_mlp2(input_to_renderer)
        input_to_renderer = self.modulate_style2(input_to_renderer, style_modulate_vec)
        out = self.graphics_renderer(input_to_renderer).view(input_to_renderer.size(0), -1)
        return out

    def get_final_output(self, z_aindep, z_adep, z_theme):
        z_content = self.merge_z_content(z_aindep, z_adep)
        if self.theme_d > 0:
            out = torch.cat([z_content, self.holistic_style_mlp(z_theme)], dim=1)
        return out

    def run_step(self, state, h, c, action, batch_size,
                read_only=False, step=0, force_input_grad=False, \
                style_h=None, force_style=None, \
                force_init_state=None, play=False, resetStyle=False, \
                logvars=None, logvars_style=None, logvars_theme=None, latent_decoder=None):
        '''
        Run dynamics engine one time step
        '''
        if not force_input_grad:
            state = state.detach()

        ## get action-independent z_aindep from linear style lstm
        z_aindep, style_h, z_aindep_kl_loss = \
                            self.run_style_lstm(state if force_init_state is None else force_init_state, force_style, style_h, \
                                [self.style_enc, self.style_enc_head, self.residual_enc_head, self.style_lstm], \
                                step, play, reset=resetStyle, logvars_style=logvars_style)

        ## run action-dependent conv lstm
        s, h, c = self.run_input_enc(force_init_state, batch_size, state, h, c)
        prev_hidden = h[0].clone()
        h, c, cur_hidden = self.engine(h, c, s, action)

        ## get action-dependent z_adep from conv lstm output
        z_adep, z_adep_kl_loss = self.run_vae_hidden(cur_hidden, self.content_enc_head, self.content_enc_head_dim, play, \
                                                                kind='content', step=step, logvars=logvars)
        ## get theme from conv lstm output
        z_theme, z_theme_kl_loss = self.run_vae_hidden(cur_hidden.view(cur_hidden.size(0), -1), \
                                                                               self.holistic_style_enc_head, self.theme_d, play, \
                                                                               kind='theme', reshape=False, step=step, logvars=logvars_theme)
        # merge all zs to get the final output
        out = self.get_final_output(z_aindep, z_adep, z_theme)

        return_dict = {}
        return_dict['prev_z'] = out
        return_dict['h'] = h
        return_dict['c'] = c
        return_dict['style_h'] = style_h # hidden state for lstm, z_aindep
        return_dict['content_hidden'] = cur_hidden # hidden state for convlstm
        return_dict['z_aindep'] = z_aindep
        return_dict['z_theme'] = z_theme
        return_dict['z_adep'] = z_adep
        return_dict['z_aindep_kl_loss'] = z_aindep_kl_loss
        return_dict['z_theme_kl_loss'] = z_theme_kl_loss
        return_dict['z_adep_kl_loss'] = z_adep_kl_loss

        return return_dict


    def run_warmup(self, states, actions, warm_up, \
                   force_style=None, \
                    force_init_state=None, latent_decoder=None):
        '''
        warm-up phase where ground-truth inputs are used
        '''
        batch_size = states[0].size(0)
        z_adep_kl_loss, z_aindep_kl_loss, z_theme_kl_loss = 0, 0, 0
        prev_z, style_h, outputs = None, None, []
        z_adeps, z_aindeps, z_themes = [], [], []

        h, c = self.engine.init_hidden(batch_size)
        h = utils.check_gpu(self.opts.gpu, h)
        c = utils.check_gpu(self.opts.gpu, c)
        for i in range(warm_up):
            input_state = states[i]
            d = self.run_step(input_state, h, c, actions[i], batch_size, \
                        step=i, force_input_grad=i==0, style_h=style_h, \
                        force_style=force_style, force_init_state=force_init_state,\
                        latent_decoder=latent_decoder)
            prev_z, h, c, style_h = d['prev_z'], d['h'], d['c'], d['style_h']
            outputs.append(d['prev_z'])
            z_adeps.append(d['z_adep'])
            z_aindeps.append(d['z_aindep'])
            z_themes.append(d['z_theme'])
            z_adep_kl_loss += d['z_adep_kl_loss']
            z_aindep_kl_loss += d['z_aindep_kl_loss']
            z_theme_kl_loss += d['z_theme_kl_loss']

            h, c = d['h'], d['c']
        warm_up_state = [h, c]
        if prev_z is None:
            prev_z = states[0] # warm_up is 0

        return_dict = {}
        return_dict['prev_z'] = prev_z
        return_dict['warm_up_state'] = warm_up_state
        return_dict['outputs'] = outputs
        return_dict['kl_losses'] = [z_adep_kl_loss, z_aindep_kl_loss, z_theme_kl_loss]
        return_dict['z_aindeps'] = z_aindeps
        return_dict['z_themes'] = z_themes
        return_dict['z_adeps'] = z_adeps
        return_dict['style_h'] = style_h
        return return_dict

    def forward(self, states, actions, warm_up, train=True, epoch=0, \
                    logvars=None, logvars_style=None, logvars_theme=None, latent_decoder=None):
        '''
        The main loop for generating a full sequence
        '''
        self.num_content_kl_added = 0
        self.num_z_aindep_kl_loss_added = 0
        self.num_theme_kl_added = 0

        batch_size = states[0].size(0)

        ##### run warm_up stage
        d = self.run_warmup(states, actions, warm_up, latent_decoder=latent_decoder)
        prev_z, warm_up_state,  outputs, kl_loss, z_adeps, style_h, z_aindeps, z_themes = \
            d['prev_z'], d['warm_up_state'], d['outputs'], d['kl_losses'], \
            d['z_adeps'], d['style_h'], d['z_aindeps'], d['z_themes']

        ##### autoregressive generation stage
        h, c = warm_up_state
        z_adep_kl_loss, z_aindep_kl_loss, z_theme_kl_loss = kl_loss
        for i in range(warm_up, len(actions) - 1):
            d = self.run_step(prev_z, h, c, actions[i], batch_size, step=i, force_input_grad=i == 0,
                        style_h=style_h, logvars=logvars, logvars_style=logvars_style,
                        logvars_theme=logvars_theme, latent_decoder=latent_decoder)
            prev_z, h, c, style_h = d['prev_z'], d['h'], d['c'], d['style_h']
            outputs.append(d['prev_z'])
            z_adeps.append(d['z_adep'])
            z_aindeps.append(d['z_aindep'])
            z_themes.append(d['z_theme'])
            z_adep_kl_loss += d['z_adep_kl_loss']
            z_aindep_kl_loss += d['z_aindep_kl_loss']
            z_theme_kl_loss += d['z_theme_kl_loss']

        response = {}
        if not self.opts.test:

            # visualize what each component does by swapping them
            if self.disentangle_style or self.theme_d > 0:
                z_adeps_concat = torch.stack(z_adeps)
                z_aindeps_concat = torch.stack(z_aindeps)
                z_themes_concat = torch.stack(z_themes) if z_themes else None
            if self.disentangle_style:
                swap_outputs = self.do_swap_z_aindep(batch_size, z_adeps_concat, z_aindeps_concat, z_themes=z_themes_concat)
                response['swap_outputs'] = swap_outputs.unbind()
            if self.theme_d > 0:
                holistic_swap_outputs = self.do_swap_z_theme(batch_size, z_adeps_concat, z_aindeps_concat, z_themes_concat)
                response['holistic_swap_outputs'] = holistic_swap_outputs.unbind()

            # kl losses
            if self.theme_d > 0:
                response['z_theme_kl_loss'] = z_theme_kl_loss / self.num_theme_kl_added
            if self.disentangle_style and self.num_z_aindep_kl_loss_added > 0:
                response['z_aindep_kl_loss'] = z_aindep_kl_loss / self.num_z_aindep_kl_loss_added
            response['z_adep_kl_loss'] = z_adep_kl_loss / (len(actions) - 1)

        response['z_adeps'] = z_adeps if self.disentangle_style else None
        response['z_aindeps'] = z_aindeps if self.disentangle_style else None
        response['outputs'] = outputs
        return response

    def update_opts(self, opts):
        self.opts = opts
        return

    def do_swap_z_aindep(self, batch_size, z_adeps, z_aindeps, z_themes=None):
        '''
        visualization purpose; see how changing z_aindep changes the output
        '''
        perm_ind = torch.randperm(batch_size, device=z_aindeps.device)
        gather_random_z_aindeps = z_aindeps[:, perm_ind, ...]
        renders = self.merge_z_content(gather_random_z_aindeps.view(batch_size * z_aindeps.shape[0], *(gather_random_z_aindeps.shape[2:])),
                                                 z_adeps.view(batch_size * z_adeps.shape[0], *(z_adeps.shape[2:])))

        if z_themes is not None and z_themes[0] is not None:
            holistic = self.holistic_style_mlp(z_themes.view(batch_size*z_themes.shape[0], *(z_themes.shape[2:])))
            swap_outputs = torch.cat([renders, holistic], dim=1)
            swap_outputs = swap_outputs.view(z_aindeps.shape[0], batch_size, *(swap_outputs.shape[1:]))
        else:
            swap_outputs = renders.view(z_aindeps.shape[0], batch_size, *(renders.shape[1:]))
        return swap_outputs

    def do_swap_z_theme(self, batch_size, z_adeps, z_aindeps, z_themes):
        '''
        visualization purpose; see how changing z_theme changes the output
        '''

        perm_ind = torch.randperm(batch_size, device=z_themes.device)
        gather_random_z_themes = z_themes[:, perm_ind, ...]

        z_theme = self.holistic_style_mlp(gather_random_z_themes.view(batch_size*z_themes.shape[0], *(gather_random_z_themes.shape[2:])))

        if self.disentangle_style:
            renders = self.merge_z_content(z_aindeps.view(batch_size * z_aindeps.shape[0], *(z_aindeps.shape[2:])),
                                                     z_adeps.view(batch_size * z_adeps.shape[0], *(z_adeps.shape[2:])))
        else:
            renders = self.graphics_renderer(z_adeps.view(batch_size * z_adeps.shape[0], *(z_adeps.shape[2:])))

        swap_outputs = torch.cat([renders, z_theme], dim=1)
        swap_outputs = swap_outputs.view(z_aindeps.shape[0], batch_size, *(swap_outputs.shape[1:]))

        return swap_outputs
