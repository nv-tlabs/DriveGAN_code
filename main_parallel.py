"""
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at the main github page.
Authors: Seung Wook Kim, Jonah Philion, Antonio Torralba, Sanja Fidler
"""

import os
import sys
import torch
import time
sys.path.append('..')
import config
import utils, visual_utils
from trainer import Trainer
import torchvision.utils as vutils
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, './data')
import dataloader
import copy
import shutil
import numpy as np

from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

def setup_multip(opts):
    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opts.distributed = n_gpu > 1
    if opts.distributed:
        print('set distributed')
        torch.cuda.set_device(opts.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()

def train_gamegan(gpu, opts):
    opts.device = 'cuda' if opts.gpu >= 0 else 'cpu'
    torch.backends.cudnn.benchmark = True

    opts = copy.deepcopy(opts)
    start_epoch = 0
    opts.img_size = (opts.img_size, opts.img_size)
    warm_up = opts.warm_up
    opts.gpu = gpu
    opts.width_mul = 1.0 if not utils.check_arg(opts, 'width_mul') else opts.width_mul
    opts.vqvae = utils.check_arg(opts, 'vqvae')
    opts.spatial_dim = 4 if not utils.check_arg(opts, 'spatial_dim') else opts.spatial_dim

    opts.do_latent = True
    latent_decoder = utils.get_latent_decoder(opts).to(opts.device)
    latent_decoder.eval()

    torch.manual_seed(opts.seed)

    # dataset ---
    print('setting up dataset')

    if opts.distributed:
        train_datasets = dataloader.get_custom_dataset(opts, set_type=0, getLoader=False, num_workers=4)
        val_datasets = dataloader.get_custom_dataset(opts, set_type=1, getLoader=False, num_workers=4)
        train_loaders, val_loaders = [], []

        def collate_fn(batch):
            batch = list(filter(lambda x: x is not None, batch))
            return torch.utils.data.dataloader.default_collate(batch)

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_datasets[0],
            shuffle=True
        )
        train_loaders = torch.utils.data.DataLoader(
            train_datasets[0],
            batch_size=opts.bs,
            sampler=train_sampler,
            drop_last=True,
            collate_fn=collate_fn)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_datasets[0],
            shuffle=False
        )
        val_loaders = torch.utils.data.DataLoader(
            val_datasets[0],
            batch_size=opts.bs,
            sampler=val_sampler,
            drop_last=False,
            collate_fn=collate_fn)
    else:
        train_loaders = dataloader.get_custom_dataset(opts, set_type=0, getLoader=True, num_workers=4)
        val_loaders = dataloader.get_custom_dataset(opts, set_type=1, getLoader=True, num_workers=4)

    # create model
    netG, netD = utils.build_models(opts)

    # choose optimizer
    optD = utils.choose_optimizer(netD, opts, opts.lrD)

    keyword = 'graphic'
    optG_temporal = utils.choose_optimizer(netG, opts, opts.lrG_temporal, exclude=keyword,
                                           model_name='optG_temporal')
    optG_graphic = utils.choose_optimizer(netG, opts, opts.lrG_graphic, include=keyword, model_name='optG_graphic')

    if opts.distributed:
        netG = nn.parallel.DistributedDataParallel(
            netG,
            device_ids=[opts.local_rank],
            output_device=opts.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
        )
        netD = nn.parallel.DistributedDataParallel(
            netD,
            device_ids=[opts.local_rank],
            output_device=opts.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
        )



    # set up logger and trainer
    logging = True if get_rank() == 0 else False
    if logging:
        logger = SummaryWriter(opts.log_dir)

    trainer = Trainer(opts,
                      netG, netD,
                      optG_temporal, optG_graphic, optD,
                      opts.LAMBDA)

    num_vis = 1 #len(train_loaders)
    save_epoch = opts.save_epoch

    def sample_data(loader):
        while True:
            for batch in loader:
                yield batch
    train_iter = sample_data(train_loaders)
    val_iter = sample_data(val_loaders)
    train_len = len(train_loaders)
    val_len = len(val_loaders)

    for epoch in range(start_epoch, opts.nep):
        print('Start epoch %d...' % epoch) if logging else None
        if epoch % save_epoch == 0 and logging:
            print('Saving checkpoint')
            if opts.distributed:
                utils.save_model(os.path.join(opts.log_dir, 'model' + str(epoch) + '.pt'), epoch, netG.module, netD.module, opts)
            else:
                utils.save_model(os.path.join(opts.log_dir, 'model' + str(epoch) + '.pt'), epoch, netG, netD, opts)
            utils.save_optim(os.path.join(opts.log_dir, 'optim' + str(epoch) + '.pt'), epoch, optG_temporal,
                             optG_graphic, optD)
        if opts.distributed:
            train_sampler.set_epoch(epoch)

        torch.cuda.empty_cache()
        log_iter = max(1,int(train_len // 10))
        write_d = 0

        times = []
        for step in range(train_len):
            it = epoch * train_len + step

            start = time.time()
            # prepare data
            states, actions, neg_actions = utils.get_data(train_iter, opts)
            states = [st.float() for st in states]
            utils.print_color('Data loading:%f' % (time.time()-start), 'yellow')

            # Generators updates
            start = time.time()

            gloss_dict, gloss, gout = \
                trainer.generator_trainstep(states, actions, warm_up=warm_up, epoch=epoch,
                                            it=it, latent_decoder=latent_decoder)
            gloss_dict = reduce_loss_dict(gloss_dict)
            gtime = time.time() - start

            # Discriminator updates
            if opts.gan_loss:
                start1 = time.time()
                dloss_dict = trainer.discriminator_trainstep(states, actions,
                                                            neg_actions, warm_up=warm_up, gout=gout,
                                                            epoch=epoch, step=step, it=it)
                dtime = time.time() - start1
                dloss_dict = reduce_loss_dict(dloss_dict)

            # Log
            if logging:
                with torch.no_grad():

                    loss_str = 'Generator [epoch %d, step %d / %d] ' % (epoch, step, train_len)
                    for k, v in gloss_dict.items():
                        if not (type(v) is float):
                            if (step % 25) == 0:
                                logger.add_scalar('losses/' + k, v.data.item(), it)
                            loss_str += k + ': ' + str(v.data.item())[:5] + ', '
                    print(loss_str)
                    utils.print_color('netG update:%f' % (gtime), 'yellow')

                    if step % 1000  == 0 and epoch %  max(1, opts.eval_epoch) == 0:
                        visual_utils.draw_output(gout, actions, neg_actions, states, opts, vutils,
                                                 logger,
                                                 it, latent_decoder=latent_decoder,
                                                 tag='trn_images')

                    if opts.gan_loss:
                        loss_str = 'Discriminator [epoch %d, step %d / %d] ' % (epoch, step, train_len)
                        for k, v in dloss_dict.items():
                            if not type(v) is float:
                                if (write_d % 25 == 0):
                                    logger.add_scalar('losses/' + k, v.data.item(), it)
                                loss_str += k + ': ' + str(v.data.item())[:5] + ', '
                        write_d += 1
                        print(loss_str)
                        utils.print_color('netD update:%f' % (dtime), 'yellow')
            del gloss_dict, gloss, gout, states, actions, neg_actions
            if opts.gan_loss:
                del dloss_dict
            torch.cuda.synchronize()
            times += [time.time() - start]

        print(f"Average iteration time: {np.mean(times)}")


        if epoch % save_epoch == 0 or epoch %  max(1, opts.eval_epoch) == 0:
            print('Validation epoch %d...' % epoch) if logging else None
            torch.cuda.empty_cache()

            max_vis = 5
            val_steps = min(100, val_len) # temporary: run small number of validation for faster training
            vis_step = max(1,val_len // max_vis)
            val_losses = {}
            for step in range(val_steps):
                it = epoch * val_len + step

                # prepare data
                states, actions, neg_actions  = utils.get_data(val_iter, opts)

                trainer.netG.eval()
                with torch.no_grad():
                    loss_dict, gloss, gout = trainer.generator_trainstep(states, actions, warm_up=warm_up,
                                                                               train=False,
                                                                               epoch=epoch,
                                                                               it=it
                                                                               )
                    if logging:
                        for key, val in loss_dict.items():
                            if key in val_losses:
                                val_losses[key] += val.item()
                            else:
                                val_losses[key] = val.item()
                        if step % vis_step == 0:
                            visual_utils.draw_output(gout, actions, neg_actions, states, opts, vutils, logger, it,
                                              latent_decoder=latent_decoder, tag='val_images')
                    del loss_dict, gloss, gout, states, actions, neg_actions
                if step % 10 == 0:
                    print(str(step)+'/'+str(val_steps))
            for key, val in val_losses.items():
                logger.add_scalar('val_losses/'+key, val / val_len, epoch)





if __name__ == '__main__':
    parser = config.init_parser()
    opts, args = parser.parse_args(sys.argv)
    setup_multip(opts)
    train_gamegan(opts.gpu, opts)
