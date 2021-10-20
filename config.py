"""
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at the main github page.
Authors: Seung Wook Kim, Jonah Philion, Antonio Torralba, Sanja Fidler
"""

from optparse import OptionParser, OptionGroup

def init_parser():
    '''
    '''
    usage = """
        Usage of this tool.
        $ python main.py [--train]
    """
    parser = OptionParser(usage=usage)
    parser.add_option('--play', action='store_true', default=False)
    parser.add_option('--port', action='store', type=int, default=8888,help='')
    parser.add_option('--log_dir', type=str, default='tmp')
    parser.add_option('--saved_model', type=str, default=None)
    parser.add_option('--saved_optim', type=str, default=None)
    parser.add_option('--data', action='store', type=str, default='pacman')

    train_param = OptionGroup(parser, 'training hyperparameters')
    train_param.add_option('--gpu', action='store', type=int, default=0)
    train_param.add_option('--local_rank', action='store', type=int, default=0)
    train_param.add_option('--num_gpu', action='store', type=int, default=1)
    train_param.add_option('--save_epoch', action='store', type=int, default=10)
    train_param.add_option('--eval_epoch', action='store', type=int, default=5)

    # optimizer
    train_param.add_option('--optimizer', action='store', type='choice', default='adam', choices=['adam', 'sgd', 'rmsprop'])
    train_param.add_option('--lrD', action='store', type=float, default=1e-4)
    train_param.add_option('--lrG_temporal', action='store', type=float, default=1e-4)
    train_param.add_option('--lrG_graphic', action='store', type=float, default=1e-4)

    # training hyperparam
    train_param.add_option('--standard_gan_loss', action='store_true', default=False)
    train_param.add_option('--warm_up', action='store', type=int, default=10)
    train_param.add_option('--bs', action='store', type=int, default=64, help='batch size')
    train_param.add_option('--nep', action='store', type=int, default=10000, help='max number of epochs')
    train_param.add_option('--img_size', action='store', type=int, default=128)
    train_param.add_option('--num_steps', action='store', type=int, default=15)
    train_param.add_option('--seed', action='store', type=int, default=10000, help='random seed')
    train_param.add_option('--disc_features', action='store_true', default=False)
    train_param.add_option('--warmup_decay_step', action='store', type=int, default=10000)
    train_param.add_option('--min_warmup', action='store', type=int, default=0)

    # losses
    train_param.add_option("--recon_loss", type=str, default="l2")
    train_param.add_option("--do_gan_loss", action="store_true", dest="gan_loss", default=True)
    train_param.add_option("--no_gan_loss", action="store_false", dest="gan_loss")
    train_param.add_option("--do_disc_features", action="store_true", dest="disc_features", default=True)
    train_param.add_option("--no_disc_features", action="store_false", dest="disc_features")
    train_param.add_option('--LAMBDA', action='store', type=float, default=1.0)
    train_param.add_option('--LAMBDA_temporal', action='store', type=float, default=10.0)
    train_param.add_option('--recon_loss_multiplier', action='store', type=float, default=0.05)
    train_param.add_option('--gen_content_loss_multiplier', action='store', type=float, default=1.0)
    train_param.add_option('--feature_loss_multiplier', action='store', type=float, default=10.0)
    train_param.add_option('--content_kl_beta', action='store', type=float, default=1.0)
    train_param.add_option('--theme_kl_beta', action='store', type=float, default=1.0)
    train_param.add_option('--style_kl_beta', action='store', type=float, default=1.0)

    # dynamics engine
    train_param.add_option("--do_input_detach", action="store_true", dest="input_detach", default=True)
    train_param.add_option("--no_input_detach", action="store_false", dest="input_detach")
    train_param.add_option('--hidden_dim', action='store', type=int, default=512)
    train_param.add_option('--width_mul', action='store', type=float, default=1.0)
    train_param.add_option('--conv_lstm_num_layer', action='store', type=int, default=2)
    train_param.add_option('--lstm_num_layer', action='store', type=int, default=1)
    train_param.add_option('--action_space', action='store', type=int, default=10)
    train_param.add_option('--disentangle_style', action='store_true', default=False)
    train_param.add_option('--continuous_action', action='store_true', default=False)
    train_param.add_option('--separate_holistic_style_dim', action='store', type=int, default=0)
    train_param.add_option('--convLSTM_hidden_dim', action='store', type=int, default=512)
    train_param.add_option('--spatial_dim', action='store', type=int, default=4)

    # discriminator
    train_param.add_option('--nfilterD', action='store', type=int, default=64)
    train_param.add_option("--do_temporal_hierarchy", action="store_true", dest="temporal_hierarchy", default=True)
    train_param.add_option("--no_temporal_hierarchy", action="store_false", dest="temporal_hierarchy")
    train_param.add_option('--nfilterD_temp', action='store', type=int, default=64)
    train_param.add_option('--config_temporal', type=int, default=18)
    train_param.add_option('--D_num_base_layer', type=int, default=3)

    # latent model
    train_param.add_option('--latent_decoder_model_path', type=str, default='')
    train_param.add_option('--latent_z_size', action='store', type=int, default=1024)

    # misc.
    train_param.add_option('--num_channel', action='store', type=int, default=3)
    train_param.add_option('--test', action='store_true', default=False)
    train_param.add_option('--initial_screen', type=str, default='')
    train_param.add_option('--recording_name', type=str, default='')
    train_param.add_option('--force_play_from_data', action='store_true', default=False)

    return parser
