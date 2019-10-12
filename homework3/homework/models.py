import torch
import torch.nn.functional as F
from .utils import _one_hot


class CNNClassifier(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, index, stride=1):
            super().__init__()
            L = [
                torch.nn.Conv2d(n_input, n_output, kernel_size=3,
                                padding=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(inplace=True),
            ]
            if index < 2:
                L.append(torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

            self.net = torch.nn.Sequential(*L)
            """
            self.downsample = None
            if stride != 1 or n_input != n_output:
                self.downsample = torch.nn.Sequential(torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride, padding=1),
                                                        torch.nn.BatchNorm2d(n_output))
                """
        def forward(self, x):
            """
            identity = x
            if self.downsample is not None:
                identity = self.downsample(identity)
            return self.net(x) + identity
            """
            return self.net(x)

    def __init__(self, layers=[64,192,384], n_input_channels=3):
        super().__init__()
        L = [torch.nn.Conv2d(n_input_channels, layers[0], kernel_size=7, padding=3, stride=2, bias=False),
            torch.nn.BatchNorm2d(layers[0]),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            ]
        c = layers[0]
        for i, l in enumerate(layers):
            L.append(self.Block(c, l, i, stride=2))
            c = l

        # add dropout before fully connected layer
        # L.append(torch.nn.Dropout())
        self.network = torch.nn.Sequential(*L)
        # final layer
        self.classifier = torch.nn.Linear(c, 6)

    def forward(self, x):
        z = self.network(x)
        z = z.mean(dim=[2,3])
        return self.classifier(z)

class FCN(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, index, stride=2):
            super().__init__()
            L = [
                torch.nn.Conv2d(n_input, n_output, kernel_size=3,
                                padding=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(inplace=True),
            ]

            self.net = torch.nn.Sequential(*L)

        def forward(self, x):
            return self.net(x)


    def __init__(self, layers=[64,192], n_input_channels=3):
        super().__init__()
        num_classes = 5
        L = [torch.nn.Conv2d(n_input_channels, layers[0], kernel_size=7, padding=3, stride=2, bias=False),
            torch.nn.BatchNorm2d(layers[0]),
            torch.nn.ReLU(inplace=True),
            ]
        c = layers[0]
        for i, l in enumerate(layers):
            L.append(self.Block(c, l, i, stride=2))
            c = l

        # output channels layer
        L.append(torch.nn.Conv2d(layers[-1], num_classes, kernel_size=1, stride=1))
        L.append(torch.nn.BatchNorm2d(num_classes))


        # final layer
        L.append(torch.nn.ConvTranspose2d(5, 5, kernel_size=(7,7), stride=(2,2), padding=(3,3), output_padding=1))
        L.append(torch.nn.BatchNorm2d(num_classes))
        L.append(torch.nn.ConvTranspose2d(5, 5, kernel_size=(3,3), stride=(2,2), padding=(1,1), output_padding=1))
        L.append(torch.nn.BatchNorm2d(num_classes))
        L.append(torch.nn.ConvTranspose2d(5, 5, kernel_size=(3,3), stride=(2,2), padding=(1,1), output_padding=1))

        self.net = torch.nn.Sequential(*L)

    def forward(self, x):
        z = self.net(x)
        H = x.shape[2]
        W = x.shape[3]
        z = z[:, :, :H, :W]
        return z


model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
