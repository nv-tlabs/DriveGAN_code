"""
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at the main github page.
Authors: Seung Wook Kim, Jonah Philion, Antonio Torralba, Sanja Fidler
"""

import torch
import utils
import numpy as np
import torch.nn.functional as F

def rescale(x):
    return (x + 1) * 0.5

def visualize_tensor(tensor, name, logger, vutils, it, kind='video'):
    tensor = rescale(tensor)
    tensor = torch.clamp(tensor, 0, 1.0)

    if kind == 'image':
        x = vutils.make_grid(
            tensor, nrow=1,
            normalize=True, scale_each=True
        )
        logger.add_image(name, x, it)
    else:
        logger.add_video(name, tensor.unsqueeze(0), it)


def write_action(actions, name, logger, it):
    s = ''
    for a in actions:
        s += str(a[:1].cpu().numpy())
    logger.add_text(name, s, it)


def draw_output(gout, actions, false_actions, states, opts, vutils, logger, it, latent_decoder=None,
                tag='images'):
    img_size = opts.img_size
    if states is not None and latent_decoder is not None:
        bs = states[0].size(0)
    else:
        bs = 0

    if actions is not None:
        write_action(actions, tag+'actions', logger, it)
    if false_actions is not None:
        write_action(false_actions, tag+'false_actions', logger, it)

    vis_st = []
    for st in states:
        vis_st.append(st[0:1])
    states_ = torch.cat(vis_st, dim=0)
    states_ = utils.run_latent_decoder(latent_decoder, states_, opts=opts)
    visualize_tensor(states_, tag + '_output/GTImage', logger, vutils, it)


    vis_st = []
    for st in gout['outputs']:
        vis_st.append(st[0:1])

    x_gen = torch.cat(vis_st, dim=0)
    x_gen = utils.run_latent_decoder(latent_decoder, x_gen, opts=opts)

    if opts.disentangle_style and 'swap_outputs' in gout:
        vis_st = []
        for st in gout['swap_outputs']:
            vis_st.append(st[0:1])
        x_gen_swap = torch.cat(vis_st, dim=0)
        x_gen_swap = utils.run_latent_decoder(latent_decoder, x_gen_swap, opts=opts)
        visualize_tensor(x_gen_swap, tag + '_output/z_aindep_SwapGenImage', logger, vutils, it)
    if opts.separate_holistic_style_dim > 0 and 'holistic_swap_outputs' in gout:
        vis_st = []
        for st in gout['holistic_swap_outputs']:
            vis_st.append(st[0:1])
        x_gen_swap = torch.cat(vis_st, dim=0)
        x_gen_swap = utils.run_latent_decoder(latent_decoder, x_gen_swap, opts=opts)
        visualize_tensor(x_gen_swap, tag + '_output/z_theme_SwapGenImage', logger, vutils, it)

    visualize_tensor(x_gen, tag + '_output/GenImage', logger, vutils, it)
