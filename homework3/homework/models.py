import torch
import torch.nn.functional as F


class CNNClassifier(torch.nn.Module):
    def __init__(self, layers=[32, 64, 128], n_input_channels=3):
        super().__init__()
        num_classes = 6
        kernel_size = 3
        stride = 2

        # init layer
        L = [torch.nn.Conv2d(n_input_channels, 32, kernel_size=7, padding=3, stride=stride),
             torch.nn.ReLU(),
             torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        c = 32

        # hidden layers
        for layer in layers:
            L.append(torch.nn.Conv2d(c, layer, kernel_size,
                                     padding=(kernel_size-1) // 2))
            L.append(torch.nn.ReLU())
            L.append(torch.nn.MaxPool2d(2*stride-1, stride, stride-1))
            c = layer
        self.network = torch.nn.Sequential(*L)

        # final layer
        self.classifier = torch.nn.Linear(c, num_classes)

    def forward(self, x):
        # normalize x before feeding it to the network

        z = self.network(x)
        z = z.mean(dim=[2,3])
        return self.classifier(z)
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        """


class FCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        raise NotImplementedError('FCN.__init__')

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
        raise NotImplementedError('FCN.forward')


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
