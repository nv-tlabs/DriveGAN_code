"""
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at the main github page.
Authors: Seung Wook Kim, Jonah Philion, Antonio Torralba, Sanja Fidler
"""

import os
import sys
import numpy as np
import torch.utils.data as data_utils
import cv2
import random
import pickle
sys.path.insert(0, './data')
sys.path.append('../../')
sys.path.append('../')
import utils
import json
import torch


def list_to_dict(l):
    d = {}
    for entry in l:
        d[entry] = 1
    return d

def gibson_get_continuous_action(pos, ori):
    actions = []
    for i in range(len(pos)-1):
        pos_diff = (pos[i+1]-pos[i])[:2]
        rot = np.array([[np.cos(ori[i][2]), -np.sin(ori[i][2])],
                        [np.sin(ori[i][2]), np.cos(ori[i][2])]])
        pos_diff = np.dot(np.transpose(rot), np.array(pos_diff))

        yaw = ori[i + 1][2] - ori[i][2]

        ## handle transitions from pi to -pi and 0 to -0 (orientation values range from (-pi, pi)
        if ori[i][2] > 0 and ori[i + 1][2] < 0:
            if abs(ori[i][2]) > 1.57 and abs(ori[i + 1][2]) > 1.57:
                yaw = (3.141593 - ori[i][2]) + (3.141593 + ori[i + 1][2])
            elif abs(ori[i][2]) < 1.57 and abs(ori[i + 1][2]) < 1.57:
                yaw = -(ori[i][2] + abs(ori[i + 1][2]))
            else:
                print('wrong orientation values?')
                exit(-1)
        elif ori[i][2] < 0 and ori[i + 1][2] > 0:
            if abs(ori[i][2]) > 1.57 and abs(ori[i + 1][2]) > 1.57:
                yaw = -((3.141593 + ori[i][2]) + (3.141593 - ori[i + 1][2]))
            elif abs(ori[i][2]) < 1.57 and abs(ori[i + 1][2]) < 1.57:
                yaw = abs(ori[i][2]) + ori[i + 1][2]
            else:
                print('wrong orientation values? 2')
                exit(-1)

        # manual scaling so that they are in more reasonable range
        actions.append([pos_diff[0] * 10, pos_diff[1] * 10, yaw* 5])
    actions = np.array(actions, dtype=np.float32)
    return actions


def get_custom_dataset(opts=None, set_type=0, force_noshuffle=False, getLoader=True, num_workers=1):

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

    dataset = []

    shuffle = True if set_type == 0 else False
    shuffle = True if opts.play else shuffle

    if force_noshuffle:
        shuffle = False

    for tmp in opts.data.split('-'):
        curdata, datadir = tmp.split(':')
        dataset.append(generic_dataset(opts, set_type=set_type, name=curdata, datadir=datadir))

    if getLoader:
        dloader = []
        for dset in dataset:
            dloader.append(data_utils.DataLoader(dset, batch_size=opts.bs,
                    num_workers=num_workers, pin_memory=True, shuffle=shuffle, drop_last=True, collate_fn=collate_fn))
        if len(dataset) == 1 and not opts.test:
            return dloader[0]
        return dloader
    else:
        return dataset




class generic_dataset(data_utils.Dataset):

    def __init__(self, opts, start=0, end=0, set_type=0, name='', datadir=''):
        self.opts = opts
        self.set_type = set_type
        self.samples = []
        self.name = name
        self.layout_memory = utils.check_arg(self.opts, 'layout_memory')
        self.continuous_action = utils.check_arg(self.opts, 'continuous_action')
        self.predict_logvar = utils.check_arg(self.opts, 'predict_logvar')
        self.learn_interpolation = utils.check_arg(self.opts, 'learn_interpolation')
        self.no_duplicate = utils.check_arg(self.opts, 'no_duplicate')
        train = True if set_type == 0 else False
        if 'gibson' in opts.data or 'carla' in opts.data:
            if 'gibson' in opts.data:
                try:
                    train_keys, val_keys, tst_keys = pickle.load(open('gibson_data_split.pkl', 'rb'))
                except:
                    train_keys, val_keys, tst_keys = pickle.load(open('../gibson_data_split.pkl', 'rb'))
            else:
                try:
                    train_keys, val_keys, tst_keys = pickle.load(open('carla_data_split.pkl', 'rb'))
                except:
                    train_keys, val_keys, tst_keys = pickle.load(open('../carla_data_split.pkl', 'rb'))

            train_keys = list_to_dict(train_keys)
            val_keys = list_to_dict(val_keys)
            tst_keys = list_to_dict(tst_keys)

            paths = []
            root_dirs = datadir
            for datadir in root_dirs.split(','):
                for fname in os.listdir(datadir):
                    cur_file = os.path.join(datadir, fname)
                    if not '.npy' in fname:
                        continue

                    key = fname.split('.')[0]
                    key = key.replace('_', '/')
                    do = False
                    if (train and key in train_keys) or (not train and key in val_keys) or (opts.test and key in tst_keys):
                        do = True
                    if not do:
                        continue
                    paths.append([key, cur_file])

        elif 'pilotnet' in opts.data:
            if '8hz' in opts.data:
                self.pilotnet_actions = pickle.load(open('8hz_all_actions.pkl', 'rb'))
                train_keys, val_keys, tst_keys = pickle.load(open('pilotnet8hz_paths_and_count.p', 'rb'))
            else:
                # 16hz
                self.pilotnet_actions = pickle.load(open('16hz_all_actions.pkl', 'rb'))
                train_keys, val_keys, tst_keys = pickle.load(open('pilotnet16hz_paths_and_count.p', 'rb'))

            paths = []
            root_dirs = datadir
            nn = 0
            random.seed(4)
            for datadir in root_dirs.split(','):

                fnames = os.listdir(datadir)
                for fname in fnames:
                    key_dict = None
                    cur_file = os.path.join(datadir, fname)

                    key = fname.split('.')[0]
                    do = False
                    is_train = False
                    if (train and (key in train_keys)):
                        do = True
                        key_dict = train_keys

                    if key in train_keys:
                        is_train = True

                    if (not train and key in val_keys):
                        do = True
                        key_dict = val_keys
                        if is_train:
                            print(key)
                            nn+= 1
                    if (opts.test and key in tst_keys):
                        do = True
                        key_dict = tst_keys

                    if key_dict is None and key in train_keys:
                        key_dict = train_keys

                    if not do:
                        continue


                    pid = key.split('_')[0]
                    if not pid in self.pilotnet_actions:
                        print(pid + ' not in pilotnet_actions file')
                        continue
                    if self.no_duplicate:
                        obj_count = 1
                    else:
                        obj_count = key_dict[key]['obj_count']
                    for _ in range(obj_count):
                        paths.append([key, cur_file])

        random.Random(4).shuffle(paths)
        if utils.check_arg(self.opts, 'num_chunk') and self.opts.num_chunk > 0:
            num_chunk = self.opts.num_chunk
            cur_ind = self.opts.cur_ind
            chunk_size = len(paths) // num_chunk
            if cur_ind == num_chunk-1:
                paths = paths[cur_ind*chunk_size:]
            else:
                paths = paths[cur_ind*chunk_size:(cur_ind+1)*chunk_size]

        tmp = np.load(paths[0][1], allow_pickle=True).item()

        opts.spatial_d = tmp['spatial_mu'].shape[1]
        opts.spatial_h = tmp['spatial_mu'].shape[2]
        opts.spatial_w = tmp['spatial_mu'].shape[3]
        opts.theme_d = tmp['theme_mu'].shape[1]
        opts.separate_holistic_style_dim = opts.theme_d
        opts.spatial_dim = opts.spatial_h

        opts.spatial_total_dim = opts.spatial_h * opts.spatial_w * opts.spatial_d


        self.samples = paths
        print('\n\n----numData: ' + str(len(paths))+ '\n\n')



    def parse_action(self, data, cur_a):
        if 'action_space' in data:
            num_actions = data['action_sapce']
        elif 'gibson' in self.name:
            num_actions = 9
            if self.continuous_action:
                action = [0] * self.opts.action_space
                for i in range(len(cur_a)):
                    action[i] = cur_a[i]
                return np.asarray(action).astype('float32'), -1
            else:
                cur_a = gibson_get_action(cur_a)

        elif 'pilotnet' in self.name:
            if self.continuous_action:
                action = [0] * self.opts.action_space
                for i in range(len(cur_a)):
                    action[i] = cur_a[i]
                return np.asarray(action).astype('float32'), -1
            else:
                print('continouse action pilotnet not supported')
                exit(-1)
        elif 'carla' in self.name:
            if self.continuous_action:
                action = [0] * self.opts.action_space
                for i in range(len(cur_a)):
                    action[i] = cur_a[i]

                return np.asarray(action).astype('float32'), -1
            else:
                cur_a = carla_get_action(cur_a[0])
                num_actions = 13
        else:
            num_actions = 10
        action = [0] * self.opts.action_space
        action[cur_a] = 1
        a_t = np.asarray(action).astype('float32')
        return a_t, num_actions

    def load_gibson(self, data):
        if self.continuous_action:
            actions = gibson_get_continuous_action(data['np_pos'], data['np_ori'])
            data['np_action'] = actions
            data['np_img_state'] = data['np_img_state'][:len(actions)]
            if 'np_img_logvar' in data:
                data['np_img_logvar'] = data['np_img_logvar'][:len(actions)]

        return data

    def load_carla(self, fn):
        data = np.load(fn, allow_pickle=True).item()
        if self.continuous_action:

            # normalize mean 0 std 1
            actions = []
            for ind in range(len(data['data'])):
                speed = (data['data'][ind]['speed'] - 18.2) / 3.62
                # + right  - left
                yaw = (data['data'][ind]['angular_velocity'][2] - (-0.40)) / 20.45
                actions.append(np.array([yaw, speed]))
            data['np_action'] = np.array(actions).astype('float32')

        return data

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        fn = self.samples[idx]
        key = fn[0]
        try:
            if 'carla' in self.opts.data:
                data = self.load_carla(fn[1])
            else:
                data = np.load(fn[1], allow_pickle=True).item()
        except:
            print('dataloader error: ')
            print(fn)
            return None

        len_episode = len(data['spatial_mu'])

        if 'gibson' in self.opts.data:
            data = self.load_gibson(data)
        elif 'pilotnet' in self.opts.data:
            if key.startswith('ind'):
                pid = key.split('#')[1]
                starting_index = key.split('#')[-1].split('.')[0]
            elif key.startswith('pn-meta'):
                pid = key.split('_')[0]
                starting_index = key.split('_')[-1].split('.')[0]
            starting_index = int(starting_index)
            action = self.pilotnet_actions[pid][starting_index:starting_index+len_episode+1]
            data['np_action'] = action

        states, actions, neg_actions, rand_actions, img_key = [], [], [], [], 'np_img_state'

        data[img_key] = np.concatenate([data['spatial_mu'].reshape(data['spatial_mu'].shape[0], self.opts.spatial_total_dim),
                                                data['theme_mu']], axis=1)


        ep_len = len_episode - self.opts.num_steps
        if self.opts.test:
            start_pt = 0  ## start from the first screen for testing
            if 'carla' in self.opts.data and self.learn_interpolation:
                start_pt = 20
        else:
            start_pt = random.randint(0, ep_len)

        i = 0
        while i < self.opts.num_steps:
            if start_pt + i >= len(data[img_key]):
                cur_s = data[img_key][len(data[img_key]) - 1]
                cur_a = data['np_action'][len(data[img_key]) - 1]
            else:
                cur_s = data[img_key][start_pt + i]
                cur_a = data['np_action'][start_pt + i]

            s_t = cur_s
            a_t, num_actions = self.parse_action(data, cur_a)

            # sample negative action within the episode
            rand_ind = random.randint(start_pt, start_pt+self.opts.num_steps - 1)
            while rand_ind == start_pt + i:
                rand_ind = random.randint(start_pt, start_pt+self.opts.num_steps - 1)
            false_a_t, _ = self.parse_action(data, data['np_action'][rand_ind])

            # save
            states.append(s_t)
            actions.append(a_t)
            neg_actions.append(false_a_t)
            i = i + 1

        del data
        return states, actions, neg_actions
