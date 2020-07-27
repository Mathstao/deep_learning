from .planner import Planner, save_model, load_model
import torch
import torchvision.transforms.functional as TF
import numpy as np


class HockeyPlayer:
    """
       Your ice hockey player. You may do whatever you want here. There are three rules:
        1. no calls to the pystk library (your code will not run on the tournament system if you do)
        2. There needs to be a deep network somewhere in the loop
        3. You code must run in 100 ms / frame on a standard desktop CPU (no for testing GPU)

        Try to minimize library dependencies, nothing that does not install through pip on linux.
    """

    """
       You may request to play with a different kart.
       Call `python3 -c "import pystk; pystk.init(pystk.GraphicsConfig.ld()); print(pystk.list_karts())"` to see all values.
    """
    kart = ""

    def __init__(self, player_id=0):
        """
        Set up a soccer player.
        The player_id starts at 0 and increases by one for each player added. You can use the player id to figure out your team (player_id % 2), or assign different roles to different agents.
        """
        all_players = ['adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley', 'kiki', 'konqi',
                       'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux', 'wilber', 'xue']
        #self.kart = all_players[np.random.choice(len(all_players))]
        #choose kart
        self.kart = 'xue'
        self.prev_loc = np.int32([0, 0])
        self.state = "kickoff"
        self.state_dict = {"kickoff": self.get_kickoff_act,
                           "reposition": self.get_reposition_act,
                           "attack": self.get_attack_act
                           }
        self.our_goal = (0, -63)
        self.their_goal = (0, 63)
        self.confident = False
        from collections import deque
        self.past_actions = deque(maxlen=5)  # empty queue of size 5
        self.past_locs = deque(maxlen=5)  # empty queue of size 5
        self.iters_since_teleport = 0
        #using planner model
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = load_model()
        self.model.eval()

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
        # TODO: use following snippet to implement reposition or position cart better
        # ori_me = self.get_vector_from_this_to_that(pos_me, front_me)
        # ori_to_item = self.get_vector_from_this_to_that(pos_me, ball_loc)
        #
        # if abs(1 - np.dot(ori_me, ori_to_item)) > 1e-4:
        #     action['steer'] = np.sign(np.cross(ori_to_item, ori_me))
        # else:
        #     action['steer'] = 0

    def ccw(self, A, B, C):  # modified simple line segment intersect function taken from: https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

        # Return true if line segments AB and CD intersect
    # simple line segment intersect function taken from: https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
    def intersect(self, A, B, C, D):
        return self.ccw(A, C, D) != self.ccw(B, C, D) and self.ccw(A, B, C) != self.ccw(A, B, D)

    # underlying math for image to local/world coordinates
    def _to_world(self, x, y, proj, view, height=0):
        W, H = 400, 300
        pv_inv = np.linalg.pinv(proj @ view)
        xy, d = pv_inv.dot([float(x) / (W / 2) - 1, 1 -
                            float(y) / (H / 2), 0, 1]), pv_inv[:, 2]
        x0, x1 = xy[:-1] / xy[-1], (xy+d)[:-1] / (xy+d)[-1]
        t = (height-x0[1]) / (x1[1] - x0[1])
        if t < 1e-3 or t > 10:
            # Project the point forward by a certain distance, if it would end up behind
            t = 10
        return t * x1 + (1-t) * x0

    # converts puck image coordinates to local/world coordinates
    def image_to_local(self, x, y, player):
        y += 40
        proj = np.array(player.camera.projection).T
        view = np.array(player.camera.view).T
        x, _, y = self._to_world(x, y, proj, view, player.kart.location[1])
        # strip ridiculous values
        if abs(x) > 39 or abs(y) > 64:
            return (99, 99)
        return (x, y)

    # called after teleporation happens; configures goal coordinates
    def set_goal_loc(self, mid_kart_loc):
        y = mid_kart_loc[-1]
        if y < 0:
            self.our_goal = (0, -63)
            self.their_goal = (0, 63)
        else:
            self.our_goal = (0, 63)
            self.their_goal = (0, -63)

    def check_teleport(self, curr_loc):
        # if we just started, we teleported
        if len(self.past_locs) == 0:
            return True
        prev_loc = self.past_locs[-1]
        if abs(np.int32(curr_loc)[0]-prev_loc[0]) > 5 or abs(np.int32(curr_loc)[2]-prev_loc[2]) >5: #locations have 3 axis

            return True
        else:
            return False

    # check if we should end the handling of the kickoff state

    def check_kickoff_over(self):
        if self.iters_since_teleport > 47:
            return True
        return False

    # check if we should move to stuck state

    def check_stuck(self, mid_kart_loc):
        # peek last value for efficient check first so we don't do the list reverse
        # TODO: calibrate this movement threshold if we're getting stuck
        movement_thresh = 0.5
        for past_loc in reversed(self.past_locs):
            x_diff = abs(past_loc[0] - mid_kart_loc[0])
            y_diff = abs(past_loc[1] - mid_kart_loc[-1])

            if x_diff > movement_thresh or y_diff > movement_thresh:
                print("check_stuck: False")
                return False
        print("check_stuck: True")
        return True

    # checks if we should attack:
    # we must be confident puck is between us and opponents goal by some padded buffer
    def check_attack(self, mid_kart_loc, puck_loc):

        # TODO: implement this function
        our_goal_y = self.our_goal[1]
        kart_y = mid_kart_loc[-1]
        puck_y = puck_loc[-1]

        distance_kart_to_our_goal = np.linalg.norm(self.to_numpy(mid_kart_loc)-np.float32(self.our_goal))
        distance_puck_our_goal = np.linalg.norm(np.float32(puck_loc)-np.float32(self.our_goal))

        behind_puck = False
        if abs(distance_kart_to_our_goal) < abs(distance_puck_our_goal):
            # meaning we are in between our goal and the puck
            behind_puck = True
            print('closer to our goal than puck is')
        # TODO: calibrate constant buffer needed between us and
        # the puck before we enter attack mode
        buffer = 1
        if behind_puck and self.confident:# and abs(kart_y - puck_y) > buffer:
            return True
        else:
            print("not ready for attack")
            return False

    # determine and set Player's state attribute

    def set_state(self, mid_kart_loc, puck_loc):
        # do the states that should be set irrespective of current position first (teleport/kickoff)
        if self.check_teleport(mid_kart_loc):
            self.state = "kickoff"
            # configure goal locations
            self.set_goal_loc(mid_kart_loc)
            self.iters_since_teleport = 0
        else:
            # if we didn't just teleport, increment this count of how long it has been
            # since the last one
            self.iters_since_teleport += 1

        # kick off state requires specific action until we've acted for a certain number
        # of iters since the last teleport
        if self.state == "kickoff" and not self.check_kickoff_over():
            self.state = "kickoff"

        # conditional on being stuck because we don't want to keep attacking the wall
        elif self.check_attack(mid_kart_loc, puck_loc) and not self.check_stuck(mid_kart_loc):
            self.state = "attack"

        # otherwise, back up towards our goal until we see the puck
        else:
            self.state = "reposition"

    # handle kickoff state
    # The agent that is aligned to the center (closer to x axis) goes full speed at the puck
    # other agent drives halfway (since we're biased towards scoring)

    def get_kickoff_act(self, mid_kart_loc, puck_loc, front_kart_loc):
        action = {'acceleration': 1, 'steer': 0, 'brake': False}
        x = mid_kart_loc[0]
        if abs(x) > 3:
            action['acceleration'] = 0.5
        return action

    # handle reposition state
    # we should backup on the vector between us and self.our_goal until we
    # are strategically postioned.
    def get_reposition_act(self, mid_kart_loc, puck_loc, front_kart_loc):
        action = {'acceleration': 1, 'steer': 0, 'brake': False}

        # can't to_numpy here because there are only 2 values
        self.their_goal = np.float32(self.their_goal)
        self.our_goal = np.float32(self.our_goal)
        mid_kart_loc = self.to_numpy(mid_kart_loc)
        front_kart_loc = self.to_numpy(front_kart_loc)
        # print("front_kart_loc",front_kart_loc)
        # print("self.our_goal",self.their_goal)
        ori_me = self.get_vector_from_this_to_that(
            mid_kart_loc, front_kart_loc)
        ori_to_item = self.get_vector_from_this_to_that(
            mid_kart_loc, self.their_goal)
        ori_to_scoring_goal = self.get_vector_from_this_to_that(
            mid_kart_loc, self.our_goal)
        # print("ori_me",ori_me)
        # print("ori_to_item",ori_to_item)
        action['brake'] = True
        action['acceleration'] = 0

        distance = np.linalg.norm(mid_kart_loc-self.their_goal)

        if distance > 10:  # if distance is too close, we'll orient ourselves not facing center sometimes
            if abs(1 - np.dot(ori_me, ori_to_item)) > 1e-4:
                action['steer'] = np.sign(np.cross(ori_to_scoring_goal, ori_me))
        else:  # remain stationary while waiting for puck to approach
            action['steer'] = np.sign(np.cross(ori_to_item, ori_me))
            action['brake'] = True
            action['acceleration'] = 0.1
        return action

    # handle attack state
    # here, we are confident the puck is between us and the opponents goal
    # we need to align our shot and drive it home

    def get_attack_act(self, mid_kart_loc, puck_loc, front_kart_loc):
        action = {'acceleration': 1, 'steer': 0, 'brake': False}

        puck_loc= np.float32(puck_loc) #can't to_numpy here because there are only 2 values
        #self.their_goal = np.float32(self.their_goal)
        mid_kart_loc = self.to_numpy(mid_kart_loc)
        front_kart_loc = self.to_numpy(front_kart_loc)
        ori_me = self.get_vector_from_this_to_that(mid_kart_loc, front_kart_loc)
        ori_to_item=self.get_vector_from_this_to_that(mid_kart_loc, puck_loc)


        if puck_loc[0]!=99: #if distance is too close, we'll orient ourselves not facing center sometimes
            if abs(1 - np.dot(ori_me, ori_to_item)) > 1e-4:
                action['steer'] = np.sign(np.cross(ori_to_item, ori_me))
        else: #remain stationary while waiting for puck to approach
            action['steer'] = 0
            action['brake']=True
            action['acceleration']=0.1
        return action

    def get_puck_known_status(self, puck_loc, player, puck_loc_world):
        if puck_loc_world == (99, 99):
            return False
        else:
            return True

    def model_controller(self, model_puck_loc, player):
        # init action dict
        action = {'acceleration': 1, 'steer': 0, 'brake': False}

        # get kart locations
        mid_kart_loc = player.kart.location
        front_kart_loc = player.kart.front

        # check if puck is out of frame (model returns negative values)
        if model_puck_loc[0] < 75 or model_puck_loc[-1] < 75:
            self.confident = False
        else:
            self.confident = True
        print("confidence",self.confident)

        # get puck location
        puck_loc = self.image_to_local(
            model_puck_loc[0], model_puck_loc[1], player)

        # determine and set state
        self.set_state(mid_kart_loc, puck_loc)

        # take action based on state
        if self.state in self.state_dict:
            # invoke appropriate state handling function
            action = self.state_dict[self.state](
                mid_kart_loc, puck_loc, front_kart_loc)
        else:
            # key in state_dict error, don't update action
            print("ERROR: invalid state")

        # keep track of past actions and locations in queue
        self.past_actions.append(action)
        self.past_locs.append(mid_kart_loc)

        # debugging info
        print(self.state)
        print(action, "\n")
        return action

    def act(self, image, player_info, state=None):
        """
        Set the action given the current image
        :param image: numpy array of shape (300, 400, 3)
        :param player_info: pystk.Player object for the current kart.
        return: Dict describing the action
        """
        """
        Your code here.
        """
        model_puck_loc = self.model(TF.to_tensor(image)[None]).squeeze(0)
        model_puck_loc = model_puck_loc.detach().cpu().numpy()
        model_action = self.model_controller(model_puck_loc, player_info)

        if state is None:
            return model_action
        else:
            return (model_action, model_puck_loc)
