from pathlib import Path
from PIL import Image
import argparse
import pystk
from time import time
import numpy as np
from . import gui


def action_dict(action):
    return {k: getattr(action, k) for k in ['acceleration', 'brake', 'steer', 'fire', 'drift']}


if __name__ == "__main__":
    soccer_tracks = {"soccer_field", "icy_soccer_field"}
    arena_tracks = {"battleisland", "stadium"}

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--track')
    # parser.add_argument('-k', '--kart', default='')
    # parser.add_argument('--team', type=int, default=0, choices=[0, 1])
    parser.add_argument('-s', '--step_size', type=float)
    parser.add_argument('--num_kart', type=int, default=2) #default 2 for a 1v1 works for all ai situation. strange movement when ai is used for larger players counts.
    parser.add_argument('--num_player', type=int, default=1)
    parser.add_argument('-p', '--play', dest='play', action='store_true') #this allows for wasd control over the cart
    parser.set_defaults(play=False)
    parser.add_argument('-v', '--visualization', type=str, choices=list(gui.VT.__members__), nargs='+',
                        default=['IMAGE'])
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--save_dir', type=Path, required=False)
    parser.add_argument('-d', '--difficulty', type=int, default=2, choices=[0, 1, 2])
    parser.add_argument('--no_images', dest='no_images', action='store_true')
    parser.set_defaults(no_images=False)
    parser.add_argument('--no_window', dest='no_window', action='store_false')
    parser.set_defaults(no_window=True)

    args = parser.parse_args()

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)
    config = pystk.GraphicsConfig.hd()
    config.screen_width = 400
    config.screen_height = 300
    config.render_window = args.no_window
    pystk.init(config)

    config = pystk.RaceConfig()

    config.num_kart = args.num_kart

    chosen_kart = 'xue'
    config.players[0].kart = chosen_kart
    config.players[0].team = 0

    if args.play:
        config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL
    else:
        config.players[0].controller = pystk.PlayerConfig.Controller.AI_CONTROL


    if args.difficulty is not None:
        config.difficulty = args.difficulty

    for i in range(args.num_player, args.num_kart):
        team =  i % 2
        if team == 0:
            kart = chosen_kart
        else:
            kart = 'tux'
        config.players.append(
            pystk.PlayerConfig(kart, pystk.PlayerConfig.Controller.AI_CONTROL, team))

    config.track = "icy_soccer_field"
    config.mode = config.RaceMode.SOCCER
    if args.step_size is not None:
        config.step_size = args.step_size

    race = pystk.Race(config)
    race.start()
    race.step()

    uis = [gui.UI([gui.VT[x] for x in args.visualization])
           for i in range(args.num_player)]

    state = pystk.WorldState()
    state.update()
    t0 = time()

    # setup data logging file number
    from os import listdir
    files = listdir(args.save_dir)
    if len(files) > 0:
        last_file = files[-1]
        underscore_split = last_file.split("_")
        ext_split = underscore_split[-1].split(".")
        start_no = int(ext_split[0])
    else:
        start_no = 0


    n = 0
    while all(ui.visible for ui in uis):
        if not all(ui.pause for ui in uis):
            race.step(uis[0].current_action)
            state.update()
            if args.verbose and config.mode == config.RaceMode.SOCCER:
                print('Score ', state.soccer.score)
                print('      ', state.soccer.ball.location)

        for ui, d in zip(uis, race.render_data):
            ui.show(d)

        if args.save_dir:
            state_dict = {}
            state_dict['puck_loc'] = state.soccer.ball.location
            for i in range(len(state.karts)):
                key = 'kart_loc_' + str(i)
                state_dict[key] = state.karts[i].location
            action = action_dict(uis[0].current_action)
            file_no = start_no + n
            #(args.save_dir / 'actions' / ('action_%06d.txt' % n)).write_text(str(action))
            (args.save_dir / ('action_%06d.txt' % file_no)).write_text(str(action))
            (args.save_dir / ('state_%06d.txt' % file_no)).write_text(str(state_dict))
            if not args.no_images:
                image = np.array(race.render_data[0].image)
                Image.fromarray(image).save(
                    args.save_dir / ('image_%06d.png' % file_no))


        # Make sure we play in real time
        n += 1
        delta_d = n * config.step_size - (time() - t0)
        if delta_d > 0:
            ui.sleep(delta_d)

    race.stop()
    del race
    pystk.clean()
