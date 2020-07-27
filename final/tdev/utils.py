import pystk
import torch
import torch.nn.functional as F
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from . import dense_transforms
from time import gmtime, strftime

DATASET_PATH = 'drive_data'

class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor()):
        from PIL import Image
        from glob import glob
        from os import path
        self.data = []
        for f in glob(path.join(dataset_path, '*.csv')):
            i = Image.open(f.replace('.csv', '.png'))
            i.load()
            self.data.append((i, np.loadtxt(f, dtype=np.float32, delimiter=',')))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = self.transform(*data)
        return data

def load_data(dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor(), num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path, transform=transform)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    """
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
                        (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)


class Player:
    def __init__(self, player, team=0):
        self.player = player
        self.team = team

    @property
    def config(self):
        return pystk.PlayerConfig(controller=pystk.PlayerConfig.Controller.PLAYER_CONTROL, kart=self.player.kart, team=self.team)

    def __call__(self, image, player_info, state):
        return self.player.act(image, player_info, state)


class Tournament:
    _singleton = None

    def __init__(self, players, screen_width=400, screen_height=300, track='icy_soccer_field'):
        assert Tournament._singleton is None, "Cannot create more than one Tournament object"
        Tournament._singleton = self

        self.graphics_config = pystk.GraphicsConfig.hd()
        self.graphics_config.screen_width = screen_width
        self.graphics_config.screen_height = screen_height
        pystk.init(self.graphics_config)

        self.race_config = pystk.RaceConfig(num_kart=len(players), track=track, mode=pystk.RaceConfig.RaceMode.SOCCER, difficulty=0)
        self.race_config.players.pop()

        self.players = players

        self.active_players = []
        for p in players:
            if p is not None:
                self.race_config.players.append(p.config)
                self.active_players.append(p)

        self.k = pystk.Race(self.race_config)

        self.k.start()
        self.k.step()

    def _to_image(self, x, proj, view):
        W, H = self.graphics_config.screen_width, self.graphics_config.screen_height
        p = proj @ view @ np.array(list(x) + [1])
        return np.array([W / 2 * (p[0] / p[-1] + 1), H / 2 * (1 - p[1] / p[-1])])

    @staticmethod
    def _point_on_track(distance, track, offset=0.0):
        """
        Get a point at `distance` down the `track`. Optionally applies an offset after the track segment if found.
        Returns a 3d coordinate
        """
        node_idx = np.searchsorted(track.path_distance[..., 1],
                                   distance % track.path_distance[-1, 1]) % len(track.path_nodes)
        d = track.path_distance[node_idx]
        x = track.path_nodes[node_idx]
        t = (distance + offset - d[0]) / (d[1] - d[0])
        return x[1] * t + x[0] * (1 - t)

    def to_numpy(self, location):
        """
        Don't care about location[1], which is the height
        """
        return np.float32([location[0], location[2]])


    def get_vector_from_this_to_that(self, me, obj, normalize=True):
        """
        Expects numpy arrays as input
        """
        vector = obj - me

        if normalize:
            return vector / np.linalg.norm(vector)

        return vector

    def test_func(self, state, player):
        x = self.to_numpy(state.soccer.ball.location)
        print(x)

    def get_action_follow_ball(self, state, player):
        action = {'acceleration': 1, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False}

        ball_loc = self.to_numpy(state.soccer.ball.location)
        pos_me = self.to_numpy(player.kart.location)
        front_me = self.to_numpy(player.kart.front)

        ori_me = self.get_vector_from_this_to_that(pos_me, front_me)
        ori_to_item = self.get_vector_from_this_to_that(pos_me, ball_loc)

        if abs(1 - np.dot(ori_me, ori_to_item)) > 1e-4:
            action['steer'] = np.sign(np.cross(ori_to_item, ori_me))
        else:
            action['steer'] = 0

        #TODO: have to change from steer = -1 or 1 to angle based on ori to ball and ori me


        if abs(action['steer']) > 0.8:
            action['acceleration'] = 0.2

        return action

    def play(self, save=None, max_frames=50):
        state = pystk.WorldState()
        if save is not None:
            import PIL.Image
            import os
            if not os.path.exists(save):
                os.makedirs(save)

        print('players',len(self.players))

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1)
        time_mark = strftime("%Y-%m-%d_%H-%M-%S", gmtime())

        for t in range(max_frames):
            print('\rframe %d' % t, end='\r')

            state.update()

            list_actions = []
            for i, p in enumerate(self.active_players):
                player = state.players[i]
                #print('k render_data size',len(self.k.render_data))
                image = np.array(self.k.render_data[i].image)

                action = pystk.Action()
                player_action,model_puck_loc = p(image, player, state)
                #Using get action
                #player_action = self.get_action_follow_ball(state, player)
                #self.test_func(state, player)

                for a in player_action:
                    setattr(action, a, player_action[a])

                list_actions.append(action)

                proj = np.array(player.camera.projection).T
                view = np.array(player.camera.view).T

                # live plot of first player
                if i == 0:
                    ax.clear()
                    ax.imshow(image)
                    #proj = np.array(player_info.camera.projection).T
                    #view = np.array(player_info.camera.view).T
                    ax.add_artist(plt.Circle(self._to_image(player.kart.location, proj, view), 2, ec='b', fill=False, lw=1.5))
                    ax.add_artist(plt.Circle(self._to_image(state.soccer.ball.location, proj, view), 2, ec='g', fill=False, lw=1.5))
                    ax.add_artist(plt.Circle(model_puck_loc, 2, ec='r', fill=False, lw=1.5))
                    #if planner:
                    #    ap = self._point_on_track(kart.distance_down_track + TRACK_OFFSET, track)
                    #    ax.add_artist(plt.Circle(self._to_image(ap, proj, view), 2, ec='g', fill=False, lw=1.5))
                    plt.pause(1e-3)

                if save is not None:
                    PIL.Image.fromarray(image).save(os.path.join(save, 'player%02d_%05d_%s.png' % (i, t, time_mark)))
                    fn_puck_loc = os.path.join(save, 'player%02d_%05d_%s.csv' % (i, t, time_mark))
                    with open(fn_puck_loc, 'w') as f1:
                        f1.write('%0.2f,%0.2f' % tuple(self._to_image(state.soccer.ball.location, proj, view)))
                    #with open(fn_kart, 'w') as f2:
                    #    f2.write('%0.2f,%0.2f' % tuple([player.kart.location[0],player.kart.location[2]]))


            s = self.k.step(list_actions)
            if not s:  # Game over
                break

        if save is not None:
            import subprocess
            for i, p in enumerate(self.active_players):
                dest = os.path.join(save, 'player%02d' % i)
                output = save + '_player%02d.mp4' % i
                subprocess.call(['ffmpeg', '-y', '-framerate', '10', '-i', dest + '_%05d.png', output])
        if hasattr(state, 'soccer'):
            return state.soccer.score
        return state.soccer_score

    def close(self):
        self.k.stop()
        del self.k
