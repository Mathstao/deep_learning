import torch
import torch.nn.functional as F


class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        return F.nll_loss(F.log_softmax(input), target)


class LinearClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        input_size = (3*64*64)
        output_size = (6)
        self.linear = torch.nn.Linear(input_size, output_size)


    def forward(self, x):
        """
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        return self.linear(x.view(x.size(0), -1))




class MLPClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = 100
        input_size = (3*64*64)
        output_size = (6)
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, output_size)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x.view(x.size(0), -1))))


model_factory = {
    'linear': LinearClassifier,
    'mlp': MLPClassifier,
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
