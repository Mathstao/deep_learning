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
    def __init__(self, n_input_channels=3):
        super().__init__()
        # block 1
        num_classes = 5
        kernel_size = 7
        padding = kernel_size // 2
        conv1 = torch.nn.Conv2d(n_input_channels, 16, kernel_size=kernel_size, padding=padding, stride=1, bias=False)
        bn1 = torch.nn.BatchNorm2d(16)
        relu1 = torch.nn.ReLU(inplace=True)


        #  block 2
        kernel_size = 3
        padding = kernel_size // 2
        conv2 = torch.nn.Conv2d(16, num_classes, kernel_size=kernel_size, padding=padding, stride=1, bias=False)
        bn2 = torch.nn.BatchNorm2d(num_classes)
        relu2 = torch.nn.ReLU(inplace=True)

        L = [conv1, bn1, relu1, conv2, bn2, relu2]
        self.net = torch.nn.Sequential(*L)
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,6,H,W))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
            if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
            convolution
        """
        z = self.net(x)
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
