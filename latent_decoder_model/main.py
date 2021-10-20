"""
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at the main github page.
Authors: Seung Wook Kim, Jonah Philion, Antonio Torralba, Sanja Fidler
"""

import argparse
import math
import os
import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import utils
from torch.utils.tensorboard import SummaryWriter
import lpips
import cv2
from model.model import Discriminator, styleVAEGAN
from dataset import ImageDataset
from lpips import networks_basic as networks
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

torch.backends.cudnn.benchmark = True

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()

def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.mean(1).sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss

def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    all_grad = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )

    path_lengths = 0
    for grad in all_grad:
        num_dim = len(grad.size())

        if num_dim == 2:
            path_lengths += torch.sqrt(grad.pow(2).sum(1))
        elif num_dim == 3:
            path_lengths += torch.sqrt(grad.pow(2).sum(2).sum(1))
        elif num_dim == 4:
            path_lengths += torch.sqrt(grad.pow(2).sum(3).sum(2).sum(1))
    path_lengths = path_lengths / len(all_grad)
    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)
    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths

def get_sample_zs(args, device, bs):
    sample = {'spatial_z': torch.randn(bs, args.z_dim, args.spatial_dim, int(args.width_mul*args.spatial_dim), device=device)}
    sample['theme_z'] = torch.randn(bs, args.theme_dim, device=device)

    return sample

def save_img(name, data, n_sample, logger, step, scale=True):
    sample = data[:n_sample]
    if scale:
        sample = sample * 0.5 + 0.5
    sample = torch.clamp(sample, 0, 1.0)
    x = utils.make_grid(
        sample, nrow=int(n_sample ** 0.5),
        normalize=False, scale_each=False
    )
    logger.add_image(name, x, step)


def get_data(loader, device, args=None):
    real_img = next(loader)
    real_img = real_img.to(device)

    return real_img

def get_disc_r1_loss(real_pred, real_img, loss_dict):
    ind = 0
    r1_loss = 0
    for rp in real_pred:
        cur_loss = d_r1_loss(rp, real_img)
        loss_dict['r1_' + str(ind)] = cur_loss
        r1_loss += cur_loss
        ind += 1
    r1_loss = r1_loss / ind
    return r1_loss

def get_disc_loss(real_pred, fake_pred, loss_dict):
    ind = 0
    d_loss = 0
    for rp, fp in zip(real_pred, fake_pred):
        cur_loss = d_logistic_loss(rp, fp)
        loss_dict['real_score'+str(ind)] = rp.detach().mean()
        loss_dict['fake_score'+str(ind)] = fp.detach().mean()

        d_loss += cur_loss
        ind += 1
    d_loss = d_loss / ind
    return d_loss

def get_g_loss(fake_pred, loss_dict):
    ind = 0
    g_loss = 0
    for fp in fake_pred:
        cur_loss = g_nonsaturating_loss(fp)
        loss_dict['g' + str(ind)] = cur_loss
        g_loss += cur_loss
        ind += 1
    g_loss = g_loss / ind
    loss_dict['g'] = g_loss
    return g_loss

def get_perceptual_loss(fake_img, real_img, loss_dict, args):
    p_loss = percept(fake_img, real_img).mean()
    if loss_dict is not None:
        loss_dict['perceptual_loss'] = p_loss
    return p_loss

def get_kl_loss(kl_losses, loss_dict, args, prefix=''):

    balanced_kl = 0
    for name, kl_loss in kl_losses.items():
        loss_dict[prefix+'_'+name+'_kl_val'] = kl_loss.mean().detach()
        if 'spatial' in name:
            multiplier = args.spatial_beta
        elif 'theme' in name:
            multiplier = args.theme_beta
        else:
            print('wrong kl value')
            exit(-1)
        balanced_kl += multiplier * kl_loss.mean()

    loss_dict['balanced_kl'] = balanced_kl
    return balanced_kl

def validation(vae, logger, val_loader, args, idx):

    vae.eval()
    discriminator.eval()
    val_losses = {}
    val_losses['val_p_loss'] = 0
    num_val = 30
    for ind_val in range(num_val):
        real_img = get_data(val_loader, device, args=args)
        out = vae(real_img)
        recon_img = out['image']
        p_loss = get_perceptual_loss(recon_img, real_img, None, args)
        val_losses['val_p_loss'] += p_loss.data.item()
        if get_rank() == 0 and ind_val % max(1, num_val // 10) == 0:
            save_img('VAL_Img/recon_img', recon_img, args.n_sample, logger, idx+ind_val)
            save_img('VAL_Img/real_img', real_img, args.n_sample, logger, idx+ind_val)
        if ind_val % 10 == 0:
            print(str(ind_val)+'/'+str(num_val))
        del out, real_img
    if get_rank() == 0:
        for key, val in val_losses.items():
            logger.add_scalar('VAL_Scalar/'+key, val / num_val, idx)
    return

def train_step(vae, vae_ema, discriminator, vae_optim, d_optim, logger, loader, args, i, vae_module, accum):
    '''
    run one step of training
    '''

    mean_path_length = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    loss_dict = {}

    vae.train()
    discriminator.train()

    if i > args.iter:
        print('Done!')
        exit(-1)

    real_img = get_data(loader, device, args=args)

    ######################### DISCRIMINATOR STEP #########################
    vae.zero_grad()
    discriminator.zero_grad()
    requires_grad(vae, False)
    requires_grad(discriminator, True)

    # run vae and discriminator
    out = vae(real_img)
    fake_pred = discriminator(out['image'])
    real_pred = discriminator(real_img)
    d_loss = get_disc_loss(real_pred, fake_pred, loss_dict)

    discriminator.zero_grad()
    (d_loss).backward()
    d_optim.step()

    # regularization
    d_regularize = i % args.d_reg_every == 0
    if d_regularize:
        real_img.requires_grad = True
        real_pred = discriminator(real_img)
        r1_loss = get_disc_r1_loss(real_pred, real_img, loss_dict)
        pair_r1_loss = 0

        discriminator.zero_grad()
        zero_for_grad_compute = 0  # DistributedDataParallel throws error if some parameter is not used
        for ind in range(len(real_pred)):
            zero_for_grad_compute += 0 * real_pred[ind][0][0]

        (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * zero_for_grad_compute).backward()
        d_optim.step()
        loss_dict['r1'] = r1_loss


    ######################### VAEGAN STEP #########################
    requires_grad(vae, True)
    vae.zero_grad()
    requires_grad(discriminator, False)
    discriminator.zero_grad()

    ## run vae and discriminator
    out = vae(real_img)
    recon_img = out['image']
    fake_pred = discriminator(recon_img)

    ## losses
    g_loss = get_g_loss(fake_pred, loss_dict)
    p_loss = get_perceptual_loss(recon_img, real_img, loss_dict, args)
    kl_loss = get_kl_loss(out['kl_losses'], loss_dict, args)
    (g_loss + args.gamma * p_loss + kl_loss).backward()
    vae_optim.step()

    # regularization
    g_regularize = i % args.g_reg_every == 0
    if g_regularize:
        vae.zero_grad()
        discriminator.zero_grad()
        z_reg = [out['spatial_z'].detach(), out['theme_z'].detach()]
        z_reg[0].requires_grad = True
        z_reg[1].requires_grad = True
        out = vae({ 'spatial_z': z_reg[0], 'theme_z': z_reg[1]}, decode_only=True)
        fake_img_reg = out['image']
        path_loss, mean_path_length, path_lengths = g_path_regularize(
             fake_img_reg, [out['spatial_latent'], out['theme_latent']], mean_path_length
        )
        weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss
        if args.path_batch_shrink:
            weighted_path_loss += 0 * fake_img_reg[0, 0, 0, 0]
        weighted_path_loss.backward()
        vae_optim.step()
        mean_path_length_avg = (
            reduce_sum(mean_path_length).item() / get_world_size()
        )
        del fake_img_reg

    loss_dict['path'] = path_loss
    loss_dict['path_length'] = path_lengths.mean()
    accumulate(vae_ema, vae_module, accum)
    loss_reduced = reduce_loss_dict(loss_dict)


    ######################### LOGGING #########################
    if get_rank() == 0:
        if i % 25 == 0:
            # log losses
            loss_str = 'step '+str(i)+': '
            for key, val in loss_reduced.items():
                loss_reduced[key] = val.mean().item()
                loss_str += key + '=' + str(loss_reduced[key])[:5] + ', '
                logger.add_scalar('Scalar/' + key, loss_reduced[key], i)
            print(loss_str)

        if i % 1000 == 0:
            vae_ema.eval()
            with torch.no_grad():
                ## log images
                save_img('Img/recon_img', recon_img, args.n_sample, logger, i)
                save_img('Img/real_img', real_img, args.n_sample, logger, i)

                ## log using ema
                out_dict = vae_ema(real_img[:args.n_sample], train=False)
                save_img('Img/EMA/recon_mu', out_dict['image'], args.n_sample, logger, i)
                out_dict = vae_ema(real_img[:args.n_sample], train=True)
                save_img('Img/EMA/recon', out_dict['image'], args.n_sample, logger, i)

                # random theme_z
                out = vae_ema(out_dict, replace_input={'theme_z': torch.randn_like(out_dict['theme_z'])}, decode_only=True)
                save_img('Img/EMA/RandomTheme', out['image'], args.n_sample, logger, i)
                # random spatial_z
                out = vae_ema(out_dict, replace_input={'spatial_z': torch.randn_like(out_dict['spatial_z'])}, decode_only=True)
                save_img('Img/EMA/RandomSpatial', out['image'], args.n_sample, logger, i)
                # random generation
                out = vae_ema(out_dict, replace_input={'spatial_z': torch.randn_like(out_dict['spatial_z']), 'theme_z': torch.randn_like(out_dict['theme_z'])}, decode_only=True)
                save_img('Img/EMA/RandomThemeSpatial', out['image'], args.n_sample, logger, i)
                del out_dict
    del out
    return

def train(args, loader, val_loader, num_val, vae, discriminator, vae_optim, d_optim, vae_ema, device, logger, percept):

    loader = sample_data(loader)
    val_loader = sample_data(val_loader)

    if args.distributed:
        vae_module = vae.module
        d_module = discriminator.module
    else:
        vae_module = vae
        d_module = discriminator
    accum = 0.5 ** (32 / (10 * 1000))

    for idx in range(args.iter):
        if idx % 1000 == 0:
            with torch.no_grad():
                validation(vae_ema, logger, val_loader, args, idx)

        i = idx + args.start_iter
        train_step(vae, vae_ema, discriminator, vae_optim, d_optim, logger, loader, args, i, vae_module, accum)

        if get_rank() == 0 and i % args.save_iter == 0 and i > 0:
            # save model
            save_dict = {
                'vae': vae_module.state_dict(),
                'vae_ema': vae_ema.state_dict(),
                'vae_optim': vae_optim.state_dict(),
                'args': args
            }
            save_dict['d'] =  d_module.state_dict()
            save_dict['d_optim'] = d_optim.state_dict()
            torch.save(save_dict, os.path.join(args.log_dir, str(i)+'.pt'))
            del save_dict


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('--iter', type=int, default=800000)
    parser.add_argument('--save_iter', type=int, default=10000)

    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--n_sample', type=int, default=6)
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--r1', type=float, default=10)
    parser.add_argument('--gamma', type=float, default=50.0)

    parser.add_argument('--path_regularize', type=float, default=2)
    parser.add_argument('--path_batch_shrink', type=int, default=2)
    parser.add_argument('--d_reg_every', type=int, default=16)
    parser.add_argument('--g_reg_every', type=int, default=4)
    parser.add_argument('--ckpt', type=str, default=None)

    parser.add_argument('--log_dir', type=str, default='/results')
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='carla')
    parser.add_argument('--n_mlp', type=int, default=8)
    parser.add_argument('--constant_input_size', type=int, default=4)
    parser.add_argument('--num_patchD', type=int, default=0)
    parser.add_argument('--theme_dim', type=int, default=128)
    parser.add_argument('--spatial_dim', type=int, default=4)
    parser.add_argument('--spatial_beta', type=float, default=2.0)
    parser.add_argument('--theme_beta', type=float, default=1.0)

    parser.add_argument('--width_mul', type=float, default=1)
    parser.add_argument('--crop_input', type=int, default=0)
    parser.add_argument('--z_dim', type=int, default=64)
    parser.add_argument('--theme_spat_dim', type=int, default=32)
    args = parser.parse_args()
    args.start_iter = 0

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = n_gpu > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()

    # net for perceptual loss
    percept = networks.PNetLin(pnet_rand=False, pnet_tune=False, pnet_type='vgg',
                                use_dropout=True, spatial=False, version='0.1', lpips=True).to(device)
    model_path = './lpips/weights/v0.1/vgg.pth'
    print('Loading vgg model from: %s' % model_path)
    percept.load_state_dict(torch.load(model_path), strict=False)

    # vae-gan model
    vae_model = styleVAEGAN
    vae = vae_model(
        args.size, n_mlp=args.n_mlp, channel_multiplier=args.channel_multiplier, args=args,
    ).to(device)
    vae_ema = vae_model(
        args.size, args.n_mlp, channel_multiplier=args.channel_multiplier, args=args
    ).to(device)
    vae_ema.eval()
    accumulate(vae_ema, vae, 0)
    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier, args=args
    ).to(device)

    # optimizers
    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
    vae_optim = optim.Adam(
        vae.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    # load ckpt if continuing training
    if args.ckpt is not None:
        print('load model:', args.ckpt)
        ckpt = torch.load(args.ckpt, map_location='cpu')

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0].split('_')[1])
        except ValueError:
            pass

        vae.load_state_dict(ckpt['vae'], strict=True)
        discriminator.load_state_dict(ckpt['d'], strict=True)
        vae_ema.load_state_dict(ckpt['vae_ema'])
        vae_optim.load_state_dict(ckpt['vae_optim'])
        d_optim.load_state_dict(ckpt['d_optim'])
        del ckpt

    # wrap for distributed training
    if args.distributed:
        vae = nn.parallel.DistributedDataParallel(
            vae,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
        )
        percept = nn.parallel.DistributedDataParallel(
            percept,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
        )
        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
        )

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

    # load training and validation datasets
    img_dataset = ImageDataset(args.path, args.dataset, args.size, train=True, args=args)
    loader = data.DataLoader(
        img_dataset,
        batch_size=args.batch,
        sampler=data_sampler(img_dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
        collate_fn=collate_fn
    )
    print('Total training dataset length: ' + str(len(img_dataset)))

    val_dataset = ImageDataset(args.path, args.dataset, args.size, train=False, args=args)
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=args.batch,
        sampler=data_sampler(val_dataset, shuffle=False, distributed=args.distributed),
        drop_last=True,
        collate_fn=collate_fn
    )
    print('Total validation dataset length: ' + str(len(val_dataset)))

    num_val = len(val_loader)
    logger = None
    if get_rank() == 0:
        logger = SummaryWriter(args.log_dir)

    train(args, loader, val_loader, num_val, vae, discriminator, vae_optim, d_optim, vae_ema, device, logger, percept)
