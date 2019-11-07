from .planner import Planner, save_model
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms

def train(args):
    from os import path
    model = Planner()
    train_logger = None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))

    import torch

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Planner().to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'det.th')))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    # load/transformm data
    import inspect
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(
        dense_transforms) if inspect.isclass(v)})
    train_data = load_data('drive_data', num_workers=4, transform=transform)

    # define loss
    mse_loss = torch.nn.MSELoss(reduction='mean')

    # start train loop
    global_step = 0
    for epoch in range(args.num_epoch):
        model.train()

        for img, target in train_data:
            img = img.to(device)
            pred, size = model(img)
            # print("Pred: ", pred[0:10])
            # print("target: ", target[0:10])
            loss = mse_loss(pred, target) * args.mse_loss
            # print("Loss: ", loss[0:10])

            if train_logger is not None and global_step % 100 == 0:
                train_logger.add_image(
                                'image', img[0], global_step)
                train_logger.add_scalar('target', target, global_step)
                train_logger.add_scalar('pred', pred, global_step)

            if train_logger is not None:
                train_logger.add_scalar('loss', loss, global_step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1


    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform',
                        default='Compose([ToTensor()])')
    parser.add_argument('-mw', '--mse_weight', type=float, default=0.01)
    args = parser.parse_args()
    train(args)
