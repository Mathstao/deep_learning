import torch
import torch.nn.functional as F
from .utils import spatial_argmax


class Planner(torch.nn.Module):
    def __init__(self):
        super().__init__()

        """
        Your code here
        """
        # H x W x 3
        self.start = torch.nn.Sequential(
                        torch.nn.Conv2d(3, 32, kernel_size=7, padding=3, stride=2),
                        torch.nn.BatchNorm2d(32),
                        torch.nn.ReLU(),
                        torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
        # H/2 x W/2 x 32
        self.layer1 = torch.nn.Sequential(
                        torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
                        torch.nn.BatchNorm2d(64),
                        torch.nn.ReLU())
                        # torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
                        # torch.nn.BatchNorm2d(64),
                        # torch.nn.ReLU())
        # H/4 x W/4 x 64
        # self.layer1_ds = torch.nn.Sequential(
        #                 torch.nn.Conv2d(32, 64, kernel_size=1, stride=2),
        #                 torch.nn.BatchNorm2d(64))

        self.layer2 = torch.nn.Sequential(
                        torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
                        torch.nn.BatchNorm2d(128),
                        torch.nn.ReLU())
                        # torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
                        # torch.nn.BatchNorm2d(128),
                        # torch.nn.ReLU())
        # H/8 x W/8 x 128
        # self.layer2_ds = torch.nn.Sequential(
        #                 torch.nn.Conv2d(64, 128, kernel_size=1, stride=2),
        #                 torch.nn.BatchNorm2d(128))

        # self.layer3 = torch.nn.Sequential(
        #                 torch.nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
        #                 torch.nn.BatchNorm2d(256),
        #                 torch.nn.ReLU(),
        #                 torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
        #                 torch.nn.BatchNorm2d(256),
        #                 torch.nn.ReLU())
        # H/16 x W/16 x 256
        # self.layer3_ds = torch.nn.Sequential(
        #                 torch.nn.Conv2d(128, 256, kernel_size=1, stride=2),
        #                 torch.nn.BatchNorm2d(256))

        # self.drop_out_layer = torch.nn.Dropout()
        # up-convolutions
        # self.layer4 = torch.nn.Sequential(
        #                 torch.nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
        #                 torch.nn.BatchNorm2d(256),
        #                 torch.nn.ReLU())
        self.layer5 = torch.nn.Sequential(
                        torch.nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                        torch.nn.BatchNorm2d(128),
                        torch.nn.ReLU())
        self.layer6 = torch.nn.Sequential(
                        torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                        torch.nn.BatchNorm2d(64),
                        torch.nn.ReLU())
        self.final = torch.nn.Sequential(
                        torch.nn.ConvTranspose2d(64, 32, kernel_size=7, stride=2, padding=3, output_padding=1),
                        torch.nn.BatchNorm2d(32),
                        torch.nn.ReLU())

        # training 3 classifiers
        self.classifer = torch.nn.Conv2d(32, 1, kernel_size=1)
        # sigmoid to restrict output between 0-1
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, img):
        """
        Your code here
        Predict the aim point in image coordinate, given the supertuxkart image
        @img: (B,3,96,128)
        return (B,2)
        """
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        H, W = img.size()[2], img.size()[3]
        #print('x',x)
        #print('x.shape',x.shape)  ## 32 x 3 x 96 x 128
        z32 = self.start(img)
        z64 = self.layer1(z32)# + self.layer1_ds(z32)
        #print('z1',z64.shape)
        z128 = self.layer2(z64)# + self.layer2_ds(z64)
        #print('z2',z128.shape)
        #z256 = self.layer3(z128) + self.layer3_ds(z128)
        #print('z3',z256.shape)
        #z256d = self.drop_out_layer(z256)
        #print('z_drop',z256d.shape)
        #z256u = self.layer4(z256d)
        #print('z4',z256u.shape)
        z128u = self.layer5(z128)
        #print('z5',z128u.shape)
        z64u = self.layer6(z128u)
        #print('z6',z64u.shape)

        z32u = self.final(z64u)
        #print('z6_plus',z32u.shape)

        #print('z7_result',self.classifer(z32u)[:, :, :H, :W].shape)
        result_class = self.classifer(z32u)[:, :, :H, :W]

        #print('model result shape',result_class.shape)
        ## 16 x 1 x 300 x 400

        # using soft argmax
        spa_argmax = spatial_argmax(torch.squeeze(result_class,1))

        #one hot with spatial argmax
        #xy_val = torch.zeros(spa_argmax.shape).float()
        #for idx, pt in enumerate(spa_argmax):
        #    x_val = (pt[0]+1.0)*63.5
        #    y_val = (pt[1]+1.0)*47.5
        #    # for each batch. [0...127][0...95]
        #    xy_val[idx][0] = x_val
        #    xy_val[idx][1] = y_val

        xy_val = (spa_argmax+1.0).to(device)
        #print('spa_argmax',spa_argmax)
        scaling_factor = torch.FloatTensor([[(W-1)/2,0.],[0.,(H-1)/2]]).to(device)
        #scaling_factor = torch.FloatTensor([[63.5,0.],[0.,44.5]]).to(device)
        xy_val = xy_val.mm(scaling_factor)

        return xy_val


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, Planner):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'planner.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = Planner()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'planner_simplified_dale_50.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    from .controller import control
    from .utils import PyTux
    from argparse import ArgumentParser


    def test_planner(args):
        # Load model
        planner = load_model().eval()
        pytux = PyTux()
        for t in args.track:
            steps = pytux.rollout(t, control, planner=planner, max_frames=1000, verbose=args.verbose)
            print(steps)
        pytux.close()


    parser = ArgumentParser("Test the planner")
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_planner(args)
