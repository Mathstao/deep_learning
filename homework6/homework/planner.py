import torch
import torch.nn.functional as F
from .utils import spatial_argmax

# change number of classes on Planner: changed to 1 for a single channel heatmap output
# this is valid assuming we don't need to map our output to steering, acceleration, etc.
# like we do in the recurrent networks. We could do this... but we'd need the game to pass
# more than one param to the controller function.
# the pytux controller function is designated as noisy_control... we could modify this
# to include a target velocity that is also outputted by our model.
# assignment says to just use steering coordinates
#TODO: figure out how to combine forward and detect properly/where to call detect and how that works with
# spatial_argmax?
#TODO: implement train loop

class Planner(torch.nn.Module):

    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                      stride=stride)
            self.c2 = torch.nn.Conv2d(
                n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2)
            self.c3 = torch.nn.Conv2d(
                n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2)
            self.b1 = torch.nn.BatchNorm2d(n_output)
            self.b2 = torch.nn.BatchNorm2d(n_output)
            self.b3 = torch.nn.BatchNorm2d(n_output)
            self.skip = torch.nn.Conv2d(
                n_input, n_output, kernel_size=1, stride=stride)

        def forward(self, x):
            return F.relu(self.b3(self.c3(F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x)))))))) + self.skip(x))

    class UpBlock(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                               stride=stride, output_padding=1)

        def forward(self, x):
            return F.relu(self.c1(x))

    # n_class = 1 because we have one channel for the heatmap: BS x H x W
    def __init__(self, layers=[16, 32, 64, 128], n_class=1, kernel_size=3, use_skip=True):
        super().__init__()
        self.input_mean = torch.Tensor([0.3521554, 0.30068502, 0.28527516])
        self.input_std = torch.Tensor([0.18182722, 0.18656468, 0.15938024])

        c = 3
        self.use_skip = use_skip
        self.n_conv = len(layers)
        skip_layer_size = [3] + layers[:-1]
        for i, l in enumerate(layers):
            self.add_module('conv%d' % i, self.Block(c, l, kernel_size, 2))
            c = l
        for i, l in list(enumerate(layers))[::-1]:
            self.add_module('upconv%d' % i, self.UpBlock(c, l, kernel_size, 2))
            c = l
            if self.use_skip:
                c += skip_layer_size[i]
        self.classifier = torch.nn.Conv2d(c, n_class, 1)
        self.size = torch.nn.Conv2d(c, 2, 1)

    def forward(self, img):
        """
        Your code here
        Predict the aim point in image coordinate, given the supertuxkart image
        @img: (B,3,96,128)
        return (B,2)
        """
        z = (img - self.input_mean[None, :, None, None].to(img.device)
             ) / self.input_std[None, :, None, None].to(img.device)
        up_activation = []
        for i in range(self.n_conv):
            # Add all the information required for skip connections
            up_activation.append(z)
            z = self._modules['conv%d' % i](z)

        for i in reversed(range(self.n_conv)):
            z = self._modules['upconv%d' % i](z)
            # Fix the padding
            z = z[:, :, :up_activation[i].size(2), :up_activation[i].size(3)]
            # Add the skip connection
            if self.use_skip:
                z = torch.cat([z, up_activation[i]], dim=1)
        cz = self.classifier(z)
        sz = self.size(z)
        cz = spatial_argmax(cz[:, 0])
        H_out = img.size()[2]
        W_out = img.size()[3]
        cz = scale_coords(cz, H_out, W_out)
        return cz

def scale_coords(coords, out_H, out_W):
    try:
        coords[:, 0] = (coords[:, 0] + 1) * out_H / 2
        coords[:, 1] = (coords[:, 1] + 1) * out_W / 2
    except:
        print("Your scale_coords broke, check the + 1 to the tensor")
    return coords

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
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'planner.th'), map_location='cpu'))
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
