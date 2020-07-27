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
    
    def __init__(self, player_id = 0):
        """
        Set up a soccer player.
        The player_id starts at 0 and increases by one for each player added. You can use the player id to figure out your team (player_id % 2), or assign different roles to different agents.
        """
        all_players = ['adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley', 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux', 'wilber', 'xue']
        #self.kart = all_players[np.random.choice(len(all_players))]
        #choose kart
        self.kart = 'xue'
        self.prev_loc = np.int32([0,0])
        self.rescue_count = 0
        self.recovery = False
        self.rescue_steer = 1
        self.current_team = 'not_sure'
        self.s_turn_left = False
        self.s_turn_right = False
        self.s_count = 0

        #using planner model
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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

    def test_func(self, state, player):
        x = self.to_numpy(state.soccer.ball.location)
        print(x)

    def get_action_follow_ball(self, state, player):
        action = {'acceleration': 1, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False}

        ball_loc = self.to_numpy(state.soccer.ball.location)
        pos_me = self.to_numpy(player.kart.location)
        front_me = self.to_numpy(player.kart.front)

        #TODO: have to change from steer = -1 or 1 to angle based on ori to ball and ori me
        if (ball_loc[0] < -10 and pos_me[1] < ball_loc[1]) or (ball_loc[0] > 10 and pos_me[1] > ball_loc[1]):
            ball_loc = np.float32([ball_loc[0]-0.5, ball_loc[1]])
        if (ball_loc[0] > 10 and pos_me[1] < ball_loc[1]) or (ball_loc[0] < -10 and pos_me[1] > ball_loc[1]):
            ball_loc = np.float32([ball_loc[0]+0.5, ball_loc[1]])

        ori_me = self.get_vector_from_this_to_that(pos_me, front_me)
        ori_to_item = self.get_vector_from_this_to_that(pos_me, ball_loc)

        if abs(1 - np.dot(ori_me, ori_to_item)) > 1e-4:
            action['steer'] = np.sign(np.cross(ori_to_item, ori_me))
        else:
            action['steer'] = 0

        #TODO: only do this when puck is not in front
        if abs(action['steer']) > 0.8:
            action['acceleration'] = 0.2

        if self.recovery == True:
            action['steer'] = -1
            action['acceleration'] = 0
            action['brake'] = True
            self.rescue_count -= 5
            if self.rescue_count < 1:
                self.recovery = False
        else:
            if self.prev_loc[0] == np.int32(pos_me)[0] and self.prev_loc[1] == np.int32(pos_me)[1]:
                self.rescue_count += 1
            else:
                if self.recovery == False:
                    self.rescue_count = 0

            if self.rescue_count > 30:
                self.recovery = True

        self.prev_loc = np.int32(pos_me)
        
        return action

    def line_direction(self, kart_loc, kart_front):
        #kart_loc => self.to_numpy(player.kart.location)
        #kart_front => self.to_numpy(player.kart.front)

        slope = (kart_loc[1] - kart_front[1])/(kart_loc[0] - kart_front[0])
        intersect = kart_loc[1] - (slope*kart_loc[0])
        facing_up_grid = kart_front[1] > kart_loc[1]
        return (slope, intersect, facing_up_grid)

    def x_intersect(self, kart_loc, kart_front):
        #kart_loc => self.to_numpy(player.kart.location)
        #kart_front => self.to_numpy(player.kart.front)

        slope = (kart_loc[1] - kart_front[1])/(kart_loc[0] - kart_front[0])
        intersect = kart_loc[1] - (slope*kart_loc[0])
        facing_up_grid = kart_front[1] > kart_loc[1]
        if slope == 0:
            x_intersect = kart_loc[1]
        else:
            if facing_up_grid:
                x_intersect = (65-intersect)/slope
            else:
                x_intersect = (-65-intersect)/slope
        return (x_intersect, facing_up_grid)

    def is_puck_in_front(self, puck_loc, threshold=2.0):
        #puck_loc => puck_loc -- model output

        x=puck_loc[0]
        return (x>(200-threshold)) and (x<(200+threshold))

    ## DO _ S _ TURN


    def model_controller(self, puck_loc, player):
        action = {'acceleration': 1, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}

        #ball_loc = self.to_numpy(state.soccer.ball.location)
        pos_me = self.to_numpy(player.kart.location)
        front_me = self.to_numpy(player.kart.front)
        kart_velocity = player.kart.velocity
        velocity_mag = np.sqrt(kart_velocity[0]**2 + kart_velocity[2]**2)

        x = puck_loc[0]     # 0-400
        y = puck_loc[1]     # 0-300

        #abs_x = np.abs(x)
        sign_x = np.sign(x)

        # clipping x and y values
        if x < 0:
            x=0
        if x > 400:
            x=400

        if y < 0:
            y=0
        if y > 300:
            y=300

        if self.current_team == 'not_sure':
            if -58<pos_me[1]<-50:
                self.current_team = 'red'
            else:
                self.current_team = 'blue'
            print('Current Team:',self.current_team)


        #print('kart id',player.kart.id,'velocity',player.kart.velocity)

        # LEAN FEATURE
        
        x_intersect, facing_up_grid = self.x_intersect(pos_me,front_me)
        lean_val = 2

        #if (175<x<225) and (100<y<120):
            #lean_val = 10
        #if (175<x<225) and (120<y<160):
            #lean_val = 5

        if -10<pos_me[0]<10:
            lean_val=0

        # facing outside goal
        #if facing_up_grid and 9<x_intersect<40 and pos_me[0]>9:
        #if facing_up_grid:
            #print('facing_up_grid',facing_up_grid)
        #print('x_intersect',x_intersect)
        #print('y loc',pos_me[1])
        #if self.current_team == 'red':
        if facing_up_grid and 9<x_intersect<40:
            #if red team
            if self.current_team == 'red':
                x += lean_val
                #print('RL_F')
            else:
                x -= lean_val
                #print('RL_B')
            #if (150<x<250) and velocity_mag > 12:
                #action['acceleration'] = 0.5
                #action['brake'] = True
                #print('RL_F')
        #if facing_up_grid and -40<x_intersect<-9 and pos_me[0]<-9:
        if facing_up_grid and -40<x_intersect<-9:
            #if red team
            if self.current_team == 'red':
                x -= lean_val
                #print('LR_F')
            else:
                x += lean_val
                #print('LR_B')
            #if (150<x<250) and velocity_mag > 12:
                #action['acceleration'] = 0.5
                #action['brake'] = True
                #print('LR_F')

        # facing inside goal
        if (not facing_up_grid) and 0<x_intersect<10:
            #if red team
            if self.current_team == 'red':
                x += lean_val
                #print('RL_B')
            else:
                x -= lean_val
                #print('RL_F')
        if (not facing_up_grid) and -10<x_intersect<0:
            #if red team
            if self.current_team == 'red':
                x -= lean_val
                #print('LR_B')
            else:
                x += lean_val
                #print('LR_F')
            

        #print('velocity_mag',velocity_mag)
        if velocity_mag > 20:
            #print('velocity_mag',velocity_mag)
            action['acceleration'] = 0.2


        
        if x < 200:
            action['steer'] = -1
        elif x > 200:
            action['steer'] = 1
        else:
            action['steer'] = 0
            # here you can put shot alignment, or narrow edging depending on where kart is and where it is facing


        if x < 50 or x > 350:
            action['drift'] = True
            action['acceleration'] = 0.2
        else:
            action['drift'] = False

        if x < 100 or x > 300:
            action['acceleration'] = 0.5


        # RECOVERY 

        if self.recovery == True:
            action['steer'] = self.rescue_steer
            action['acceleration'] = 0
            action['brake'] = True
            self.rescue_count -= 2
            #print('rescue_count',self.rescue_count)
            # no rescue if initial condition
            if self.rescue_count < 1 or ((-57<pos_me[1]<57 and -7<pos_me[0]<1) and velocity_mag < 5):
                self.rescue_count = 0
                self.recovery = False
        else:
            if self.prev_loc[0] == np.int32(pos_me)[0] and self.prev_loc[1] == np.int32(pos_me)[1]:
                self.rescue_count += 5
            else:
                if self.recovery == False:
                    self.rescue_count = 0

            #if self.rescue_count > 30 or pos_me[1]>65 or pos_me[1]<-65:
            #if self.rescue_count > 30 or ((y>200) and (50<x<350)):
            if self.rescue_count<2:
                if x<200:
                    self.rescue_steer = 1
                else:
                    self.rescue_steer = -1
            if self.rescue_count > 30 or (y>200):
                #if x<200:
                    #self.rescue_steer = 1
                #else:
                    #self.rescue_steer = -1
                # case of puck near bottom left/right
                if velocity_mag > 10:
                    self.rescue_count = 30
                    self.rescue_steer = 0
                else:
                    self.rescue_count = 20
                self.recovery = True

        self.prev_loc = np.int32(pos_me)
        
        return action

    def _to_image(self, x, proj, view):
        W, H = 400, 300
        p = proj @ view @ np.array(list(x) + [1])
        return np.array([W / 2 * (p[0] / p[-1] + 1), H / 2 * (1 - p[1] / p[-1])])

    def act(self, image, player_info, state=None):
        """
        Set the action given the current image
        :param image: numpy array of shape (300, 400, 3)
        :param player_info: pystk.Player object for the current kart.
        return: Dict describing the action
        """
        #print('kart id, state.soccer.loc',player_info.kart.id,state.soccer.ball.location)
        #print('kart id, loc',player_info.kart.id,player_info.kart.location)
        #action = {'acceleration': 0, 'brake': True, 'drift': False, 'nitro': False, 'rescue': True, 'steer': np.random.uniform(-0.5,0.5)}
        """
        Your code here.
        """

        #action = self.get_action_follow_ball(state, player_info)

        #using model

        model_puck_loc = self.model(TF.to_tensor(image)[None]).squeeze(0)
        model_puck_loc = model_puck_loc.detach().cpu().numpy()
        #print('model_puck_loc',model_puck_loc)
        model_action = self.model_controller(model_puck_loc, player_info)

        #return (action,model_puck_loc)
        #return model_action
        return (model_action,model_puck_loc)
