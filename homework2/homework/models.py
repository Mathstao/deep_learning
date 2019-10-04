import torch

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
        for layer in layers:
            L.append(torch.nn.Conv2d(c, layer, kernel_size, padding=(kernel_size-1) // 2))
            L.append(torch.nn.ReLU())
            L.append(torch.nn.MaxPool2d(2*stride-1, stride, stride-1))
            c = layer
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(c, num_classes)

    def forward(self, x):
        z = self.network(x)
        z = z.mean(dim=[2,3])
        return self.classifier(z)


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, CNNClassifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn.th'))
    raise ValueError("model type '%s' not supported!"%str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = CNNClassifier()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th'), map_location='cpu'))
    return r
