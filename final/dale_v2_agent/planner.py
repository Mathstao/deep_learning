import torch
from .utils import spatial_argmax


class Planner(torch.nn.Module):
    def __init__(self, channels=[16, 32, 64, 128]):
        super().__init__()

        conv_block = lambda c, h: [torch.nn.BatchNorm2d(h), torch.nn.Conv2d(h, c, 7, 2, 3), torch.nn.ReLU(True)]
        upconv_block = lambda c, h: [torch.nn.BatchNorm2d(h), torch.nn.ConvTranspose2d(h, c, 4, 2, 1),
                                     torch.nn.ReLU(True)]

        h, _conv, _upconv = 3, [], []
        for c in channels:
            _conv += conv_block(c, h)
            h = c

        for c in channels[:-3:-1]:
            _upconv += upconv_block(c, h)
            h = c

        _upconv += [torch.nn.BatchNorm2d(h), torch.nn.Conv2d(h, 1, 1, 1, 0)]

        self._conv = torch.nn.Sequential(*_conv)
        self._upconv = torch.nn.Sequential(*_upconv)

        #self._mean = torch.FloatTensor([0.30025885, 0.2711953, 0.2725873])
        #self._std = torch.FloatTensor([0.1056248, 0.11337676, 0.13114084])

        self._mean = torch.FloatTensor([0.4519, 0.5590, 0.6204])
        self._std = torch.FloatTensor([0.0012, 0.0018, 0.0020])

    def forward(self, img):
        """
        Your code here
        Predict the aim point in image coordinate, given the supertuxkart image
        @img: (B,3,96,128)
        return (B,2)
        """
        img = (img - self._mean[None, :, None, None].to(img.device)) / self._std[None, :, None, None].to(img.device)
        h = self._conv(img)
        x = self._upconv(h)
        return (1 + spatial_argmax(x.squeeze(1))) * torch.as_tensor([img.size(3) - 1, img.size(2) - 1]).float().to(
            img.device)


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
