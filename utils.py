"""
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at the main github page.
Authors: Seung Wook Kim, Jonah Philion, Antonio Torralba, Sanja Fidler
"""

import torch
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
from torch import distributions
import math
import random
import numpy as np
import os


def run_latent_decoder(latent_decoder, input, opts=None, return_all_outputs=False):
    decoder_input = [input]

    z_first, z_last = input[:, :opts.spatial_total_dim], input[:, opts.spatial_total_dim:]
    z_first = z_first.view(input.size(0), -1, opts.spatial_h, opts.spatial_w)
    bs = 4
    num_chunk = int(np.ceil(z_first.size(0) / bs))
    imgs = []
    for i in range(num_chunk):
        out = latent_decoder({'spatial_z':z_first[i*bs:(i+1)*bs],
                            'theme_z': z_last[i*bs:(i+1)*bs]}, decode_only=True)
        imgs.append(out['image'])
    output = torch.cat(imgs, dim=0)

    return output


def get_latent_decoder(opts):
    from latent_decoder_model.model.model import styleVAEGAN
    img_size = opts.img_size[0]
    latent_z_size = opts.latent_z_size
    latent_decoder_model_path = opts.latent_decoder_model_path

    if opts.gpu >= 0:
        saved_ckpt = torch.load(latent_decoder_model_path)
    else:
        saved_ckpt = torch.load(latent_decoder_model_path, map_location=torch.device('cpu'))

    opts.DO_PLAIN_GAN =  False

    saved_args = saved_ckpt['args']
    vae_model = styleVAEGAN
    model = vae_model(
        saved_args.size, n_mlp=saved_args.n_mlp, channel_multiplier=saved_args.channel_multiplier, args=saved_args,
    )
    model.load_state_dict(saved_ckpt['vae_ema'], strict=False)

    model.eval()
    model = model.cpu()

    return model

def save_model(fname, epoch, netG, netD, opts):
    outdict = {'epoch': epoch, 'netG': netG.state_dict(), 'netD': netD.state_dict(), 'opts': opts}
    torch.save(outdict, fname)

def save_optim(fname, epoch, optG_temporal, optG_graphic, optD):
    outdict = {'epoch': epoch, 'optG_temporal': optG_temporal.state_dict(), 'optG_graphic': optG_graphic.state_dict(), 'optD': optD.state_dict()}
    torch.save(outdict, fname)

def adjust_learning_rate(opt, lr):
    for param_group in opt.param_groups:
        param_group['lr'] = lr

def choose_optimizer(model, options, lr=None, exclude=None, include=None, model_name=''):
    try:
        wd = options.wd
    except:
        wd = 0.0

    if lr == None:
        lr = options.lr

    if type(model) is list:
        params = model
    else:
        params = model.parameters()
        if exclude is not None:
            params = []
            for name, W in model.named_parameters():
                if type(exclude) is list:
                    excluded = False
                    for exc in exclude:
                        if exc in name:
                            excluded = True
                            print(model_name + ', Exclude: ' + name)
                            break
                    if not excluded:
                        params.append(W)
                        print(model_name + ', Include: ' + name)
                elif not exclude in name:
                    params.append(W)
                    print(model_name + ', Include: ' + name)
                else:
                    print(model_name + ', Exclude: ' + name)
        if include is not None:
            params = []
            for name, W in model.named_parameters():
                if type(include) is list:
                    for inc in include:
                        if inc in name:
                            params.append(W)
                            print(model_name + ', Include: ' + name)
                            break
                elif include in name:
                    params.append(W)
                    print(model_name + ', Include: ' + name)


    if options.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=lr,
                weight_decay=wd)
    elif options.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=lr,
                weight_decay=wd, betas=(0.0, 0.9))
    elif options.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(params, lr=lr, alpha=0.95, eps=0.01, momentum=0.9)
    else:
        raise RuntimeError('invalid oprimizer type')

    return optimizer

def build_models(opts, tmp_get_old=False):
    from simulator_model.dynamics_engine import EngineGenerator as Generator
    from simulator_model.discriminator import Discriminator

    # Build models
    generator = Generator(
        opts
    )
    discriminator = Discriminator(
        opts,
        nfilter=opts.nfilterD
    )

    if opts.gpu is not None and not opts.gpu < 0 :
        return generator.to(opts.device), discriminator.to(opts.device)
    else:
        return generator, discriminator

def weights_init(m):
    if isinstance(m, MyConvo2d):
        if m.conv.weight is not None:
            if m.he_init:
                init.kaiming_uniform_(m.conv.weight)
            else:
                init.xavier_uniform_(m.conv.weight)
        if m.conv.bias is not None:
            init.constant_(m.conv.bias, 0.0)
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)

def copy_weights(source, target):
    target.data = source.data
    return

from termcolor import colored
def print_color(txt, color):
    ''' print <txt> to terminal using colors
    '''
    print(colored(txt, color))
    return

def check_arg(opts, arg):
    v = vars(opts)
    if arg in v:
        if type(v[arg]) == bool:
            return v[arg]
        else:
            return True
    else:
        return False

def check_gpu(gpu, *args):
    '''
    '''
    if gpu == None or gpu < 0:
        if isinstance(args[0], dict):
            d = args[0]
            var_dict = {}
            for key in d:
                var_dict[key] = Variable(d[key])
            if len(args) > 1:
                return [var_dict] + check_gpu(gpu, *args[1:])
            else:
                return [var_dict]
        if isinstance(args[0], list):
            return [Variable(a) for a in args[0]]
        # a list of arguments
        if len(args) > 1:
            return [Variable(a) for a in args]
        else:
            return Variable(args[0])

    else:
        if isinstance(args[0], dict):
            d = args[0]
            var_dict = {}
            for key in d:
                var_dict[key] = Variable(d[key]).to('cuda')
            if len(args) > 1:
                return [var_dict] + check_gpu(gpu, *args[1:])
            else:
                return [var_dict]
        if isinstance(args[0], list):
            return [Variable(a).to('cuda') for a in args[0]]
        # a list of arguments
        if len(args) > 1:
            return [Variable(a).to('cuda') for a in args]
        else:
            return Variable(args[0]).to('cuda')

def get_data(data_iters, opts, get_rand=False):

    tmp_states, tmp_actions, tmp_neg_actions = [], [], []
    states, actions, neg_actions = [], [], []

    if type(data_iters) is list:
        for data_iter in data_iters:
            s, a, na = data_iter.next()
            tmp_states.append(s)
            tmp_actions.append(a)
            tmp_neg_actions.append(na)
    else:
        s, a, na = next(data_iters)
        tmp_states.append(s)
        tmp_actions.append(a)
        tmp_neg_actions.append(na)

    for j in range(len(tmp_states[0])): # over time steps
        gs, ga, gna = [], [], []
        for k in range(len(tmp_states[0][0])): # over batches
            for i in range(len(tmp_states)): # over data type
                gs.append(tmp_states[i][j][k])
                ga.append(tmp_actions[i][j][k])
                gna.append(tmp_neg_actions[i][j][k])
        states.append(torch.stack(gs, dim=0))
        actions.append(torch.stack(ga, dim=0))
        neg_actions.append(torch.stack(gna, dim=0))

    states = [check_gpu(opts.gpu, a) for a in states]
    actions = [check_gpu(opts.gpu, a) for a in actions]
    neg_actions = [check_gpu(opts.gpu, a) for a in neg_actions]

    return states, actions, neg_actions

def load_state_dict(self, state_dict):
    import torch.nn as nn
    own_state = self.state_dict()
    for name, param in state_dict.items():

        if name not in own_state:
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ' + name + ' NOT LOADED')
            continue

        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        try:
            own_state[name].copy_(param)
            print('++++++++++++++++++++++++++++++ ' + name + ' LOADED')
        except:
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ' + name + ' NOT LOADED')
            print(param.size())
            print(own_state[name].size())
            continue

def compute_grad2(d_out, x_in, allow_unused=False, batch_size=None, gpu=0, ns=1):
    if d_out is None:
        return check_gpu(gpu, torch.FloatTensor([0]))
    if batch_size is None:
        batch_size = x_in.size(0)

    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True,
        allow_unused=allow_unused
    )[0]
    # import pdb; pdb.set_trace();

    grad_dout2 = grad_dout.pow(2)
    # xassert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1) * (ns * 1.0 / 6)
    return reg

def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

def load_my_state_dict(self, state_dict):
    import torch.nn as nn
    own_state = self.state_dict()
    print('now')
    for name, param in own_state.items():
        print(name)
        if name not in state_dict:
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ' + name + ' NOT EXIST IN SAVED MODEL CKPT')
            continue
    print('load')
    for name, param in state_dict.items():
        print(name)
        name = name.replace('module.', '')

        if name not in own_state:
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ' + name + ' NOT EXIST IN CURRENT CODE MODEL FILE')
            continue
        print(name)
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        try:
            own_state[name].copy_(param)
        except:
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ' + name + ' NOT LOADED')
            print(param.size())
            print(own_state[name].size())
            continue

def init_config_model_for_play():
    import config
    import sys
    import numpy as np
    from trainer import Trainer
    parser = config.init_parser()
    opts, args = parser.parse_args(sys.argv)
    force_play_from_data = opts.force_play_from_data
    saved_model = opts.saved_model
    initial_screen = opts.initial_screen
    seed = opts.seed
    port = opts.port
    log_dir = opts.log_dir
    gpu = opts.gpu
    latent_decoder_model_path = opts.latent_decoder_model_path
    recording_name = opts.recording_name

    # create model
    opts = torch.load(
        opts.saved_model,
        map_location='cpu')['opts']
    opts.seed = seed
    opts.port = port
    opts.log_dir = log_dir
    opts.play = True
    opts.bs = 1
    opts.gpu = gpu
    opts.saved_model = saved_model
    opts.initial_screen = initial_screen
    opts.recording_name = recording_name
    opts.latent_decoder_model_path= latent_decoder_model_path
    opts.force_play_from_data = force_play_from_data

    warm_up = opts.warm_up
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    opts.width_mul = 1.0 if not check_arg(opts, 'width_mul') else opts.width_mul
    opts.spatial_dim = 4 if not check_arg(opts, 'spatial_dim') else opts.spatial_dim
    if not check_arg(opts, 'spatial_h'):
        opts.spatial_h = opts.spatial_dim
    if not check_arg(opts, 'spatial_w'):
        opts.spatial_w = int(opts.spatial_dim*opts.width_mul)
    if not check_arg(opts, 'theme_d'):
        opts.theme_d = opts.separate_holistic_style_dim
    if not check_arg(opts, 'spatial_d'):
        opts.spatial_d = (opts.latent_z_size - opts.theme_d) // int(opts.spatial_dim*opts.spatial_dim*opts.width_mul)
    opts.device = 'cuda'

    if opts.gpu != -1:
        torch.cuda.manual_seed(opts.seed)

    # create model
    netG, _ = build_models(opts)
    trainer = Trainer(opts,
                      netG, None,
                      None, None, None,
                      opts.LAMBDA)
    assert opts.saved_model != None, 'Empty saved model'

    # load the weights
    print('loading netG')
    load_my_state_dict(trainer.netG,
                       torch.load(
                           opts.saved_model,
                           map_location='cpu')['netG'] )

    print('Models loaded')

    latent_decoder = get_latent_decoder(opts)
    latent_decoder = latent_decoder.cuda(0)
    return opts, trainer, gpu, latent_decoder
