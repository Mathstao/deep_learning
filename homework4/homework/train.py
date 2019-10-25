import torch
import numpy as np
import math

from .models import Detector, save_model
from .utils import load_detection_data, ConfusionMatrix
from . import dense_transforms
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    model = Detector()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = Detector().to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(
            path.join(path.dirname(path.abspath(__file__)), 'det.th')))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    # load and transform data
    import inspect
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(
        dense_transforms) if inspect.isclass(v)})
    train_data = load_detection_data(
        'dense_data/train', num_workers=4, transform=transform)
    # convert val images to heatmaps
    transform = dense_transforms.Compose(
        [dense_transforms.ToTensor(), dense_transforms.ToHeatmap()])
    valid_data = load_detection_data('dense_data/valid', num_workers=4, transform=transform)

    loss = torch.nn.BCEWithLogitsLoss()

    global_step = 0
    for epoch in range(args.num_epoch):
        model.train()
        for img, det_map, size_map in train_data:
            img, det_map = img.to(device), det_map.to(device).long()
            logit = model(img)
            loss_val = loss(logit, det_map.float())
            """
            gamma = 2.0
            loss_val = loss_val * torch.pow((1 - torch.exp(-1 * loss_val)), gamma)
            """
            loss_val = loss_val * (1 - torch.exp(-1 * loss_val))
            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1
        """
        model.eval()
        for img, det_map, size_map in valid_data:
            img, det_map = img.to(device), det_map.to(device).long()
            logit = model(img)
        """
        print("Completed Epoch: ", epoch)

    save_model(model)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=1)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-2)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform',
                        default='Compose([ToTensor(), ToHeatmap()])')
    # Winner: python3 -m solution.train_cnn -t "Compose([ColorJitter(0.5, 0.3, 0.2), RandomHorizontalFlip(), ToTensor()])" -lr 1e-2  --log_dir log/res_k3_flip_norm_color_deeper_nodrop_long/ -n 150
    args = parser.parse_args()
    train(args)
