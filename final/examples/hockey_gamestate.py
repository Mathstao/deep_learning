from time import time

import time as pause

import pystk
import numpy as np
import matplotlib.pyplot as plt

from . import gui


def to_numpy(location):
    """
    Don't care about location[1], which is the height
    """
    return np.float32([location[0], location[2]])


def get_vector_from_this_to_that(me, obj, normalize=True):
    """
    Expects numpy arrays as input
    """
    vector = obj - me

    if normalize:
        return vector / np.linalg.norm(vector)

    return vector


if __name__ == "__main__":
    config = pystk.GraphicsConfig.hd()
    config.screen_width = 128
    config.screen_height = 96

    pystk.init(config)

    config = pystk.RaceConfig()
    config.track = "icy_soccer_field"
    config.mode = config.RaceMode.SOCCER
    config.step_size = 0.1
    config.num_kart = 2
    config.players[0].kart = "tux"
    config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL
    config.players[0].team = 0
    config.players.append(
            pystk.PlayerConfig("", pystk.PlayerConfig.Controller.AI_CONTROL, 1))

    race = pystk.Race(config)
    race.start()

    uis = [gui.UI([gui.VT['IMAGE']])]

    state = pystk.WorldState()
    t0 = time()
    n = 0

    ax = plt.gcf().add_subplot(3, 3, 9)

    prev_loc = np.int32([0,0])
    rescue_count = 0
    recovery = False

    while all(ui.visible for ui in uis):
        if not all(ui.pause for ui in uis):
            race.step(uis[0].current_action)
            state.update()

        pos_ball = to_numpy(state.soccer.ball.location)
        pos_ai = to_numpy(state.karts[1].location)
        pos_me = to_numpy(state.karts[0].location)

        # Look for the closest pick up (bomb, gift, etc..).
        closest_item = to_numpy(state.items[0].location)
        closest_item_distance = np.inf

        for item in state.items:
            item_norm = np.linalg.norm(
                    get_vector_from_this_to_that(pos_me, to_numpy(item.location), normalize=False))

            if item_norm < closest_item_distance:
                closest_item = to_numpy(item.location)
                closest_item_distance = item_norm

        #soccer ball location
        ball_loc = to_numpy(state.soccer.ball.location)
        if (ball_loc[0] < -10 and pos_me[1] < ball_loc[1]) or (ball_loc[0] > 10 and pos_me[1] > ball_loc[1]):
            ball_loc = np.float32([ball_loc[0]-0.5, ball_loc[1]])
        if (ball_loc[0] > 10 and pos_me[1] < ball_loc[1]) or (ball_loc[0] < -10 and pos_me[1] > ball_loc[1]):
            ball_loc = np.float32([ball_loc[0]+0.5, ball_loc[1]])

        # Get some directional vectors.
        front_me = to_numpy(state.karts[0].front)
        ori_me = get_vector_from_this_to_that(pos_me, front_me)
        ori_to_ai = get_vector_from_this_to_that(pos_me, pos_ai)
        #ori_to_item = get_vector_from_this_to_that(pos_me, closest_item)
        ori_to_item = get_vector_from_this_to_that(pos_me, ball_loc)

        # Turn towards the item to pick up. Not very good at turning.
        uis[0].current_action.acceleration = 1

        if abs(1 - np.dot(ori_me, ori_to_item)) > 1e-4:
            uis[0].current_action.steer = np.sign(np.cross(ori_to_item, ori_me))
            #if uis[0].current_action.steer > 0.8:
            #    uis[0].current_action.acceleration = 0.4

        #uis[0].current_action.steer = 0.25

        # prev loc for rescue

        if recovery == True:
            uis[0].current_action.steer = -1
            uis[0].current_action.acceleration = 0
            uis[0].current_action.brake = True
            rescue_count -= 5
            if rescue_count < 1:
                recovery = False
        else:
            if prev_loc[0] == np.int32(pos_me)[0] and prev_loc[1] == np.int32(pos_me)[1]:
                rescue_count += 1
            else:
                if recovery == False:
                    rescue_count = 0
                    uis[0].current_action.rescue = False

            if rescue_count > 30:
                #uis[0].current_action.rescue = True
                recovery = True

        prev_loc = np.int32(pos_me)


        print('me loc',pos_me)
        print('front_me',front_me)
        print('ori_me',ori_me)
        print('ball_loc',ball_loc)
        print('ori_to_item',ori_to_item)
        print('np cross',np.cross(ori_to_item, ori_me))
        print('count',rescue_count)
        print('')

        #pause.sleep(1.5)

        # Live plotting. Sorry it's ugly.
        ax.clear()
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)

        ax.plot(pos_me[0], pos_me[1], 'r.')                 # Current player is a red dot.
        ax.plot(pos_ai[0], pos_ai[1], 'b.')                 # Enemy ai is a blue dot.
        ax.plot(pos_ball[0], pos_ball[1], 'co')             # The puck is a cyan circle.
        ax.plot(closest_item[0], closest_item[1], 'kx')     # The target picked up is a black x.

        # Plot lines of where I am facing, and where the enemy is in relationship to me.
        ax.plot([pos_me[0], pos_me[0] + 10 * ori_me[0]], [pos_me[1], pos_me[1] + 10 * ori_me[1]], 'r-')
        ax.plot([pos_me[0], pos_me[0] + 10 * ori_to_ai[0]], [pos_me[1], pos_me[1] + 10 * ori_to_ai[1]], 'b-')

        # Live debugging of scalars. Angle in degrees to the target item.
        ax.set_title('%.2f' % (np.degrees(np.arccos(np.dot(ori_me, ori_to_item)))))

        # Properties of the karts. Overall useful to see what properties you have.
        # print(dir(state.karts[0]))

        for ui, d in zip(uis, race.render_data):
            ui.show(d)

        # Make sure we play in real time
        n += 1
        delta_d = n * config.step_size - (time() - t0)
        if delta_d > 0: ui.sleep(delta_d)

    race.stop()
    del race
    pystk.clean()
