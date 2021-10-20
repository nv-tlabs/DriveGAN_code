"""
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at the main github page.
Authors: Seung Wook Kim, Jonah Philion, Antonio Torralba, Sanja Fidler
"""

from tornado import web, ioloop
import base64
from io import BytesIO
import json
import os
import sys
import torch
import cv2
import torchvision
import random
import time
from torchvision.utils import save_image
from torch.nn.modules.upsampling import Upsample
sys.path.append('..')
import torch.nn.functional as F
import config
import utils
from trainer import Trainer
from visual_utils import rescale

sys.path.insert(0, './data')

import numpy as np



opts, trainer, gpu, latent_decoder = utils.init_config_model_for_play()
staterecords = {}
fnames = [opts.initial_screen]

########## configure web
class NoCacheStaticFileHandler(web.StaticFileHandler):
    def set_extra_headers(self, path):
        self.set_header('Cache-Control',
                        'no-store, no-cache, must-revalidate, max-age=0')

class MainHandler(web.RequestHandler):
    def get(self):
        self.render("./frontend/demo.html")

    def post(self):
        global next_id
        cmd = json.loads(self.request.body.decode('utf8').replace("'", '"'))
        print('received', cmd)
        if cmd['web_id'] == 'nothing_yet':
            print('making a new portal!', next_id)
            staterecords[next_id] = StateRecord()
            cmd['web_id'] = next_id
            next_id = (next_id + 1) % 5

        if cmd['cmd'] == 'save_frame':
            staterecords[cmd['web_id']].save_vectors(cmd['filename'])
        elif cmd['cmd'] == 'save_state_vec':
            staterecords[cmd['web_id']].save_state(cmd['filename'])
        elif cmd['cmd'] == 'load_npy':
            theme_names, part_names = staterecords[cmd['web_id']].load_npy(cmd['filename'])
            self.write(json.dumps({'web_id': cmd['web_id'], 'theme_names': theme_names, 'part_names': part_names}))
        elif cmd['cmd'] == 'change_from_list':
            new_screen = staterecords[cmd['web_id']].change_from_list(cmd['kind'], cmd['name'])
            if type(new_screen) == list:
                new_screen = new_screen[-1]
            self.write(json.dumps({'new_screen': pil_to_b64(new_screen).decode('utf8').replace("'", '"'), 'web_id': cmd['web_id']}))
        elif cmd['cmd'] == 'load_screen':
            img = staterecords[cmd['web_id']].load_screen(cmd['filename'])
            self.write(json.dumps({'web_id': cmd['web_id'], 'img': pil_to_b64(img).decode('utf8').replace("'", '"')}))
        elif cmd['cmd'] == 'resume':
            staterecords[cmd['web_id']].is_stopped = False
            staterecords[cmd['web_id']].screen_being_edited = None
            staterecords[cmd['web_id']].cur_selected_part = None
        elif cmd['cmd'] == 'start_recording':
            if staterecords[cmd['web_id']].do_recording:
                staterecords[cmd['web_id']].do_recording = False
                staterecords[cmd['web_id']].save_seq = [{}]
            else:
                staterecords[cmd['web_id']].do_recording = True

            staterecords[cmd['web_id']].recording_name = cmd['filename']
            self.write(json.dumps({'web_id': cmd['web_id']}))
        elif cmd['cmd'] == 'stop_recording' or (cmd['cmd'] == 'change_grid' and staterecords[cmd['web_id']].is_stopped):
            staterecords[cmd['web_id']].is_stopped = True
            if cmd['cmd'] == 'stop_recording':
                # 'STOP' action
                stop_action = []
                if 'carla' in opts.data:
                    stop_action = [-5, 0]
                elif 'gibson' in opts.data:
                    stop_action = [0, 0]
                elif 'pilotnet' in opts.data:
                    stop_action = [-3, 0]
                else:
                    print('\n\nNot implemented\n\n')
                    exit(-1)

                self.write(json.dumps({'web_id': cmd['web_id'], 'stop_speed': stop_action[0], 'stop_yaw': stop_action[1]}))
                staterecords[cmd['web_id']].stop_recording()
            if cmd['cmd'] == 'change_grid':
                imgs = staterecords[cmd['web_id']].change_grid(cmd['x'], cmd['y'])
                imgs = [pil_to_b64(img).decode('utf8').replace("'", '"') for img in imgs]

                self.write(json.dumps({'web_id': cmd['web_id'], 'imgs': imgs}))

        elif cmd['cmd'] in ['reset', 'next_frame']:
            if 'key' in cmd:
                staterecords[cmd['web_id']].update_action(cmd['key'])
            if cmd['cmd'] == 'reset':
                print(staterecords.keys())
                if len(fnames) == 0:
                    selected = None
                else:
                    selected = fnames[random.randint(0, len(fnames) - 1)]
                    print('File name: ' + selected)
                imgs = staterecords[cmd['web_id']].reset(selected)

            result = {'web_id': cmd['web_id']}
            if cmd['cmd'] == 'next_frame':
                imgs, gt_info = staterecords[cmd['web_id']].stepper()
                if gt_info is not None:
                    result['gt_img'] = pil_to_b64(gt_info['gt_img']).decode('utf8').replace("'", '"')

                    result['gt_speed'] = gt_info['gt_action'][0]
                    result['gt_yaw'] = gt_info['gt_action'][1]
                    result['optimized_speed'] = gt_info['optimized_action'][0]
                    result['optimized_yaw'] = gt_info['optimized_action'][1]
            imgs = [pil_to_b64(img).decode('utf8').replace("'", '"') for img in imgs]
            result['imgs'] = imgs
            self.write(json.dumps(result))
        elif cmd['cmd'] == 'change_hscene':
            new_screen = staterecords[cmd['web_id']].reset_z_theme()
            self.write(json.dumps({'new_screen': pil_to_b64(new_screen).decode('utf8').replace("'", '"'), 'web_id': cmd['web_id']}))
        elif cmd['cmd'] == 'change_scene':
            new_screen = staterecords[cmd['web_id']].reset_z_aindep()
            self.write(json.dumps({'new_screen': pil_to_b64(new_screen).decode('utf8').replace("'", '"'), 'web_id': cmd['web_id']}))
        elif cmd['cmd'] == 'change_content':
            new_screen = staterecords[cmd['web_id']].reset_z_adep()
            self.write(json.dumps({'new_screen': pil_to_b64(new_screen).decode('utf8').replace("'", '"'), 'web_id': cmd['web_id']}))
        elif cmd['cmd'] == 'stop_recording':
            staterecords[cmd['web_id']].stop_recording()
            self.write(json.dumps({'web_id': cmd['web_id']}))
        elif not  staterecords[cmd['web_id']].is_stopped:
            if  staterecords[cmd['web_id']].dir_imgs is not None:
                result = {'web_id': cmd['web_id']}
                img, action = staterecords[cmd['web_id']].play_from_directory()
                result['dir_img'] = pil_to_b64(img).decode('utf8').replace("'", '"')
                result['speed'] = str(action[0])
                result['yaw'] = str(action[1])
                if 'pilotnet' in opts.data:
                    kind = 'pilotnet'
                elif 'gibson' in opts.data:
                    kind = 'gibson'
                elif 'carla' in opts.data:
                    kind = 'carla'
                result['kind'] = kind
                self.write(json.dumps(result))

            else:
                if staterecords[cmd['web_id']].prev_screen is not None:
                    staterecords[cmd['web_id']].update_action(cmd['cmd'])
                    imgs, gt_info = staterecords[cmd['web_id']].stepper()
                    result = {'web_id': cmd['web_id']}
                    if gt_info is not None:
                        result['gt_img'] = pil_to_b64(gt_info['gt_img'].decode('utf8').replace("'", '"'))
                        result['gt_speed'] = str(gt_info['gt_action'][0])
                        result['gt_yaw'] = str(gt_info['gt_action'][1])

                        if 'optimized_action' in gt_info:
                            result['optimized_speed'] = str(gt_info['optimized_action'][0])
                            result['optimized_yaw'] = str(gt_info['optimized_action'][1])
                    imgs = [pil_to_b64(img).decode('utf8').replace("'", '"') for img in imgs]
                    result['imgs'] = imgs
                    self.write(json.dumps(result))

def serve():
    port = opts.port
    loop = ioloop.IOLoop.instance()
    app = web.Application([
        (r"/", MainHandler),
        (r"/(.*)", NoCacheStaticFileHandler, {
            "path":
                os.path.join(os.path.dirname(__file__), "./frontend/")})
    ], debug=True)
    print('view @ http://localhost:{}'.format(port))
    app.listen(port)
    loop.start()

def pil_to_b64(img):
    buffer = BytesIO()
    img.save(buffer, 'JPEG')
    return base64.b64encode(buffer.getvalue())


class StateRecord(object):
    def __init__(self):
        if opts.img_size[0] < 256:
            self.upsample = Upsample(scale_factor=4, mode='nearest')
        else:
            self.upsample = Upsample(scale_factor=2, mode='nearest')
        self.opts = opts
        self.trainer = trainer
        self.trainer.netG.eval()
        self.is_stopped = False
        self.do_recording= False
        self.h = None
        self.c = None
        self.prev_z = None
        self.update_action('stop')
        self.time_step = 0
        self.warm_up = 0
        self.step = 0
        self.resetStyle = False
        self.reset_counter = 0
        self.force_style = None
        self.force_init_state = None
        self.force_theme = None
        self.cur_theme = None
        self.screen_being_edited = None
        self.cur_content = None
        self.part_vector = None
        self.recording_name = ''
        self.selected_themes = []
        self.selected_parts = []
        self.themes = None
        self.parts = None
        self.cur_selected_part = None
        self.optimized_seq = None
        self.gt_seq = None
        self.save_seq = [{}]
        self.prev_screen = None
        self.optimized_logvars = None
        self.optimized_logvars_style = None
        self.optimized_logvars_theme = None
        self.freeAction = 1
        self.firstImage = None
        self.dir_imgs = None

    def reset_save_arrays(self):
        self.latent_save = []
        self.style_save = []
        self.content_save = []
        self.theme_save = []
        self.keep_screens = []
        self.keep_actions = []
        self.reset_counter = 0

    def reset_lstm(self, states, actions, force_state_len=False, no_reset=False):
        with torch.no_grad():
            self.step = 0
            self.reset_save_arrays()

            d = self.trainer.netG.run_warmup(
                states, actions, len(states) if force_state_len else self.warm_up,
                force_style=self.force_style, force_init_state=self.force_init_state)
            warm_up_state = d['warm_up_state']

            self.step = len(states) if force_state_len else self.warm_up
            self.time_step = len(states) if force_state_len else self.warm_up
            self.trainer.netG.num_residual_kl_loss_added = 0
            self.style_h =  d['style_h']
            self.prev_rnn_state = warm_up_state
            self.prev_screen = d['prev_z']

            if self.opts.initial_screen == 'rand' and not force_state_len and not no_reset:
                self.reset_z_theme(init=True)
                self.reset_z_aindep(init=True)
                self.reset_z_adep()
                d['prev_z'] = self.force_init_state
                self.prev_screen = d['prev_z']
            else:
                self.force_theme = None
                self.force_style = None
                self.force_init_state = None

            prev_z = utils.run_latent_decoder(latent_decoder, d['prev_z'], opts=opts)
            prev_z = torch.clamp(prev_z, -1.0, 1.0)
            img = rescale(prev_z)

        self.keep_screens = [states[self.warm_up - 1]]
        self.keep_actions = [actions[self.warm_up - 1]]
        return img

    def change_grid(self, x, y):
        edit_screen = self.screen_being_edited
        if self.screen_being_edited is None:
            edit_screen = self.prev_screen

        content = edit_screen[:, :-trainer.netG.theme_d]
        theme = edit_screen[:, -trainer.netG.theme_d:]
        mask = torch.zeros([content.size(0), 1, self.opts.spatial_h, self.opts.spatial_w])
        content_reshape = content.view(content.size(0), -1, self.opts.spatial_h, self.opts.spatial_w)

        new_content = torch.randn_like(content_reshape)
        if self.cur_selected_part is not None:
            for s_ind in range(len(x)):
                new_content[0, :, int(y[s_ind]), int(x[s_ind])] = self.cur_selected_part
                mask[:, :, int(y[s_ind]), int(x[s_ind])] = 1.0
        else:
            sx = int(x * self.opts.spatial_w)
            sy = int(y * self.opts.spatial_h)
            mask[:, :, sy, sx] = 1.0
            self.part_vector = new_content[:, :, sy, sx].cpu().numpy()

        mask = mask.to(content.device)
        new_content = new_content * mask + content_reshape * (1 - mask)
        new_content = new_content.view(content.size(0), -1)

        self.force_init_state = torch.cat([new_content, theme], dim=1)
        self.screen_being_edited = self.force_init_state

        prev_z = utils.run_latent_decoder(latent_decoder, self.force_init_state, opts=opts)
        prev_z = torch.clamp(prev_z, -1.0, 1.0)
        img = rescale(prev_z)
        imgs = [img]

        init_state_save = self.force_init_state.clone()

        # 'STOP' action
        if 'carla' in self.opts.data:
            self.current_action[:, 0] = 0
            self.current_action[:, 1] = -5
        elif 'gibson' in self.opts.data:
            self.current_action[:, 0] = 0
            self.current_action[:, 1] = 0
            self.current_action[:, 2] = 0
        elif 'pilotnet' in self.opts.data:
            self.current_action[:, 0] = -3
            self.current_action[:, 1] = 0
        else:
            print('\n\nNot implemented\n\n')
            exit(-1)

        result = self.reset_lstm([self.screen_being_edited], [self.current_action], no_reset=True)
        result = [torchvision.transforms.ToPILImage()(self.upsample(result).clamp(0, 1)[0].cpu())]
        # adapt model to the new screen for 3 more time steps
        for i in range(3):
            # 'STOP' action
            if 'carla' in self.opts.data:
                self.current_action[:, 0] = 0
                self.current_action[:, 1] = -5
            elif 'gibson' in self.opts.data:
                self.current_action[:, 0] = 0
                self.current_action[:, 1] = 0
                self.current_action[:, 2] = 0
            elif 'pilotnet' in self.opts.data:
                self.current_action[:, 0] = -3
                self.current_action[:, 1] = 0
            else:
                print('\n\nNot implemented\n\n')
                exit(-1)
            self.force_init_state = self.screen_being_edited
            self.prev_screen = self.screen_being_edited
            result, _ = self.stepper()

        return result

    def reset_z_theme(self, init=False):
        '''
        randomize z_theme
        '''
        if self.trainer.netG.theme_d <= 0:
            self.force_theme = None
            return

        self.reset_save_arrays()
        self.force_theme = utils.check_gpu(opts.gpu, torch.randn(1, trainer.netG.theme_d))
        if not init:
            return self.reset_zs(self.cur_style, self.cur_content, self.force_theme, reset_engine=False)


    def reset_z_aindep(self, init=False, style=None):
        '''
        randomize z_aindep
        '''

        self.reset_save_arrays()
        if not self.trainer.netG.disentangle_style:
            self.force_style = None
            return

        # reset hidden state
        self.style_h = None

        new_style = utils.check_gpu(opts.gpu, torch.randn(1, trainer.netG.opts.hidden_dim)) if style is None else style
        if not init:
            # reset the image input with new style
            return self.reset_zs(new_style, self.cur_content, self.cur_theme, reset_engine=False)
        else:
            # random vector used to randomly initialize the initial img
            self.force_style = new_style

    def reset_z_adep(self):
        '''
        randomize z_adep
        '''

        self.reset_save_arrays()
        z_aindep = self.force_style if self.force_style is not None else self.cur_style
        z_theme = self.force_theme if self.force_theme is not None else self.cur_theme

        z_adep = utils.check_gpu(opts.gpu, torch.randn(1, self.trainer.netG.opts.hidden_dim))
        return self.reset_zs(z_aindep, z_adep, z_theme, reset_engine=False)

    def reset_zs(self, z_aindep, z_adep, z_theme, reset_engine=False):
        '''
        get the new screen with corresponding zs
        '''
        with torch.no_grad():
            if reset_engine:
                self.reset_engine_hidden()
            self.force_init_state = self.trainer.netG.get_final_output(z_aindep, z_adep, z_theme)
            new_screen = utils.run_latent_decoder(latent_decoder, self.force_init_state, opts=opts)
            new_screen = torch.clamp(new_screen, -1.0, 1.0)
            new_screen = rescale(new_screen)
            return torchvision.transforms.ToPILImage()(new_screen.clamp(0, 1)[0].cpu())

    def reset(self, initial_screen):
        '''
        reset everything
        '''
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if initial_screen is None:
            tmp = utils.check_gpu(opts.gpu, torch.randn(1, 512))
            if add_mean_std:
                states = [(latent_decoder.style(tmp) - latent_mean) / latent_std]
            else:
                states = [latent_decoder.style(tmp)]
            actions = [torch.FloatTensor([0] * 10).unsqueeze(0)]
        else:
            states = [torch.FloatTensor(0).unsqueeze(0)]
            actions = [torch.FloatTensor([0] * 10).unsqueeze(0)]
        states = [utils.check_gpu(opts.gpu, a) for a in states]
        actions = [utils.check_gpu(opts.gpu, a) for a in actions]
        img = self.reset_lstm(states, actions)

        return [torchvision.transforms.ToPILImage()(img[0].cpu())]

    def update_action(self, cmd):
        '''
        update the current action given from the ui
        '''
        if cmd == 'stop':
            self.ac = [0, 0]
        elif 'carla' in self.opts.data:
            self.ac = [cmd['yaw'], cmd['speed']]
        elif 'pilotnet' in self.opts.data:
            self.ac = [cmd['speed'], -2 * cmd['yaw']]
        elif 'gibson' in self.opts.data:
            self.ac = [cmd['speed'], 0, -1 * cmd['yaw']]
        else:
            pass

        a_t = [0] * self.opts.action_space
        for i in range(len(self.ac)):
            a_t[i] = self.ac[i]
        a_t = np.asarray(a_t).astype('float32')

        print(a_t)
        self.current_action = utils.check_gpu(opts.gpu, torch.FloatTensor(a_t).unsqueeze(0))

    def reset_engine_hidden(self):
        h, c = self.trainer.netG.engine.init_hidden(1)
        h = utils.check_gpu(self.opts.gpu, h)
        c = utils.check_gpu(self.opts.gpu, c)
        self.prev_rnn_state[0] = h
        self.prev_rnn_state[1] = c

    def stepper(self):
        start_time = time.time()

        with torch.no_grad():
            state_z = None
            base_imgs = []

            if self.do_recording and not 'output' in self.save_seq[0]:
                self.firstImage = utils.run_latent_decoder(latent_decoder, self.prev_screen, opts=opts)
                self.firstImage = torch.clamp(self.firstImage, -1.0, 1.0)

                self.firstImage = rescale(self.firstImage)
                self.save_seq[0]['output'] = np.uint8(np.transpose(self.firstImage.clamp(0, 1)[0].cpu() * 255, (1,2,0)))

            prev_screen_save = torch.clone(self.prev_screen)

            num_reset = opts.num_steps // 3
            do_reset, resetStyle = False, False
            self.reset_counter += 1
            if self.reset_counter > num_reset:
                do_reset = True
                self.reset_counter = 0

            if do_reset and len(self.keep_screens) > 0:
                self.reset_lstm(self.keep_screens[-3:], self.keep_actions[-3:], force_state_len=True)
                self.resetStyle = True

            print(self.step)
            if not self.freeAction and self.optimized_seq is not None and self.step < len(self.optimized_seq['optimized_actions']):
                a_t = [0] * self.opts.action_space
                if 'gibson' in self.opts.data and len(self.optimized_seq['optimized_actions'][0]) < 3:
                    a_t[0] = self.optimized_seq['optimized_actions'][self.step][0]
                    a_t[2] = self.optimized_seq['optimized_actions'][self.step][1]
                else:
                    for i in range(len(self.ac)):
                        a_t[i] = self.optimized_seq['optimized_actions'][self.step][i]
                a_t = np.asarray(a_t).astype('float32')
                self.current_action = utils.check_gpu(opts.gpu, torch.FloatTensor(a_t).unsqueeze(0))

            d = trainer.netG.run_step(
                self.prev_screen, self.prev_rnn_state[0],
                self.prev_rnn_state[1], self.current_action, 1,
                step=self.time_step, style_h=self.style_h, \
                play=True,
                force_style=self.force_style, \
                force_init_state=self.force_init_state, resetStyle=self.resetStyle, \
                logvars=self.optimized_logvars, logvars_style=self.optimized_logvars_style, logvars_theme=self.optimized_logvars_theme)
            self.prev_screen, self.style_h = d['prev_z'], d['style_h']
            self.resetStyle=False
            self.force_style = None
            self.force_init_state = None
            self.force_theme = None
            self.cur_theme = d['z_theme']
            self.cur_style = d['z_aindep']
            self.cur_content = d['z_adep']
            self.prev_rnn_state = [d['h'], d['c']]
            self.keep_screens.append(prev_screen_save)
            self.keep_actions.append(self.current_action)
            self.step += 1
            self.time_step += 1

            prev_z = utils.run_latent_decoder(latent_decoder, self.prev_screen, opts=opts)
            img = rescale(prev_z)
            imgs = [img]

        result = []
        for i in range(len(imgs)):
            result.append(torchvision.transforms.ToPILImage()(self.upsample(imgs[i]).clamp(0, 1)[0].cpu()))
        print('\n\nTook: ' + str(time.time() - start_time) + '\n\n')

        cur_action = self.current_action.cpu().numpy()[0][:2]
        if 'gibson' in self.opts.data:
            cur_action = np.array([self.current_action.cpu().numpy()[0][0], self.current_action.cpu().numpy()[0][2]])

        if 'carla' in self.opts.data :
            op_tmp = -cur_action[0]
            cur_action[0] = cur_action[1]
            cur_action[1] = op_tmp

        if self.do_recording:
            seq = {'output': np.uint8(np.transpose(imgs[0].clamp(0, 1)[0].cpu() * 255, (1,2,0)))}
            self.save_seq.append(seq)
            self.save_seq[-2]['cur_action'] = cur_action

        return result, None

    def change_from_list(self, kind, name):
        if kind == 'theme':
            vec = self.themes[name]
            return self.reset_zs(self.cur_style, self.cur_content, vec, reset_engine=False)
        elif kind == 'part' and name != 'random':
            vec = self.parts[name]
            h = self.part_h_coords[name]
            w = self.part_w_coords[name]
            self.cur_selected_part = vec
            return self.change_grid(w, h)


    def save_state(self, filename):
        np.save(os.path.join('/home/seung/Projects/simulator_v2/simulator/init_screen', filename), self.prev_screen.cpu().numpy())

    def save_vectors(self, filename):
        self.selected_themes.append(self.cur_theme.cpu().numpy())
        vectors = {}
        vectors['theme'] = np.squeeze(np.array(self.selected_themes), axis=1)
        if self.part_vector is not None:
            self.selected_parts.append(self.part_vector)
            vectors['selected_parts'] = np.squeeze(np.array(self.selected_parts), axis=1)
        np.save(filename, vectors)

    def load_npy(self, filename):
        if os.path.isdir(filename):
            parentdir = filename
            fnames = os.listdir(filename)
            filename = ''
            for fn in fnames:
                if fn.endswith('.npy'):
                    filename += os.path.join(parentdir, fn) + ','
            filename = filename[:-1]

        themes, theme_names = {}, []
        parts, part_names, part_h_coords, part_w_coords = {}, [], {}, {}
        for f in filename.split(','):
            data = np.load(f, allow_pickle=True).item()
            info = os.path.basename(f).split('.')[0].split('_')
            name = info[0]
            kind = info[1]
            if kind == 'theme' or kind == 'holistic_style':
                try:
                    themes[name] = utils.check_gpu(opts.gpu, torch.FloatTensor(np.mean(data['theme'], axis=0)).unsqueeze(0))
                except:
                    themes[name] = utils.check_gpu(opts.gpu, torch.FloatTensor(np.mean(data['holistic_style'], axis=0)).unsqueeze(0))
                theme_names.append(name)
            elif kind == 'part':
                h, w = info[2].split('-')
                parts[name] = utils.check_gpu(opts.gpu, torch.FloatTensor(np.mean(data['selected_parts'], axis=0)).unsqueeze(0))
                part_names.append(name)
                part_h_coords[name] = h
                part_w_coords[name] = w

            else:
                print('wrong kind in load_npy')
                exit(-1)
        theme_names.sort()
        part_names.sort()
        self.themes = themes
        self.parts = parts
        self.part_h_coords = part_h_coords
        self.part_w_coords = part_w_coords
        return theme_names, part_names

    def load_screen(self, name):
        initizl_z = np.load(name)
        return self.reset_lstm([utils.check_gpu(opts.gpu, torch.FloatTensor(initizl_z))], [None], no_reset=True)

    def stop_recording(self):
        pass

if __name__ == '__main__':
    next_id = 0
    serve()
