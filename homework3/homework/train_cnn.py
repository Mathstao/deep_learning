from .models import CNNClassifier, save_model
from .utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb
from os import path


def train(args, model):
    #model = CNNClassifier()

    # setup GPU
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # load model
    if args.continue_training:
        model.load_state_dict(torch.load(
            path.join(path.dirname(path.abspath(__file__)), '%s.th' % args.model)))

    # setup tensorboard logging
    train_logger = tb.SummaryWriter('logs/train')
    valid_logger = tb.SummaryWriter('logs/valid')

    # define optimizer and loss function
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    loss = torch.nn.CrossEntropyLoss()

    # get data
    train_data = load_data('../data/train')
    valid_data = load_data('../data/valid')

    global_step = 0
    # begin training
    for epoch in range(args.num_epoch):
        model.train()
        loss_vals, acc_vals, vacc_vals = [], [], []
        for img, label in train_data:
            img, label = img.to(device), label.to(device)
            logit = model(img)
            loss_val = loss(logit, label)
            acc_val = accuracy(logit, label)

            loss_vals.append(loss_val.detach().cpu().numpy())
            acc_vals.append(acc_val.detach().cpu().numpy())

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            train_logger.add_scalar(
                "Train Loss", loss_val, global_step=global_step)

            global_step += 1

        avg_loss = sum(loss_vals) / len(loss_vals)
        avg_acc = sum(acc_vals) / len(acc_vals)
        train_logger.add_scalar(
            "Train Accuracy", avg_acc, global_step=global_step)

        model.eval()
        for img, label in valid_data:
            img, label = img.to(device), label.to(device)
            vacc_vals.append(
                accuracy(model(img), label).detach().cpu().numpy())
        avg_vacc = sum(vacc_vals) / len(vacc_vals)
        valid_logger.add_scalar(
            "Validation Accuracy", avg_vacc, global_step=global_step)
        print('epoch %-3d \t loss = %0.3f \t acc = %0.3f \t val acc = %0.3f' %
              (epoch, avg_loss, avg_acc, avg_vacc))

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-4)
    parser.add_argument('-c', '--continue_training', action='store_true')
    args = parser.parse_args()
    train(args)
