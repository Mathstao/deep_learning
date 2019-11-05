import pystk
import math
import numpy as np


def control(aim_point, current_vel):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in local coordinate frame
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    action = pystk.Action()
    """
    1) figure out how to tailor acceleration to more narrow tracks
    2) when to use nitro
    3) optimize hyper parameters
    """

    # optimizations
    # steering angle
    # accleration
    # hyperparameters
    target_vel = 25
    steer_factor = 0.9
    drift_thresh = 0.9
    accel_factor = 0.35
    nitro_thresh = 0.1
    start_steering = 0.05

    # Your code here
    # Hint: Use action.brake to True/False to brake (optionally)
    # Hint: Use action.steer to turn the kart towards the aim_point, clip the steer angle to -1..1
    x = aim_point[0]
    # handle case where we're going the wrong way
    theta = np.arctan(x)
    theta /= math.pi / 2
    if abs(theta) < start_steering:
        action.steer = 0
    else:
        action.steer = theta * steer_factor
    # Hint: You may want to use action.drift=True for wide turns (it will turn faster)
    if abs(theta) > drift_thresh:
        action.drift = True
    else:
        action.drift = False

    # Hint: Use action.acceleration (0..1) to change the velocity.
    # Try targeting a target_velocity (e.g. 20).
    vel_ratio = current_vel / target_vel
    if current_vel > target_vel:
        action.acceleration = 0
    elif abs(theta) > 0.9:
        action.acceleration = 0.1
    else:
        # accelerate proportionally to target speed
        #action.acceleration = 1 - vel_ratio - abs(theta) * accel_factor
        action.acceleration = 1 - vel_ratio * accel_factor
    if abs(theta) < nitro_thresh:
        action.nitro = True
    else:
        action.nitro = False
    return action


if __name__ == '__main__':
    from .utils import PyTux
    from argparse import ArgumentParser


    def test_controller(args):
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(steps)
        pytux.close()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
