from .models import ClassificationLoss, model_factory, save_model
from .utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('device =', device)

def train(args):
    # setup logger
    log_dir = "logs/"
    train_logger = tb.SummaryWriter(log_dir+'/model/train', flush_secs=1)
    valid_logger = tb.SummaryWriter(log_dir+'/model/valid', flush_secs=1)

    # create model
    model = model_factory[args.model]()
    learning_rate = 0.01
    n_epochs = 100
    batch_size = 128
    # create loss
    loss = ClassificationLoss()
    # create optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    path_root = "data/"
    training_generator = load_data(path_root + "train", batch_size=batch_size)
    validation_generator = load_data(path_root + "valid", batch_size=batch_size)


    # for epoch in range(args.ne):
    global_step = 0
    for epoch in range(n_epochs):

        # load each batch with load_data
        print("Loading training data batch for epoch", epoch, "...")
        # train_dataloader = load_data(path_root + "train", batch_size=args.bs)
        for local_batch, local_labels in training_generator:
            # make prediction
            train_pred = model.forward(local_batch)

            # evaluate prediction
            l = loss.forward(train_pred, local_labels)

            # log loss
            train_logger.add_scalar('Train Loss', l, global_step=global_step)

            # compute accuracy
            acc = accuracy(train_pred, local_labels)

            # log accuracy
            train_logger.add_scalar('Train Accuracy', acc, global_step=global_step)

            # zero out the gradient so it doesn't carry
            optimizer.zero_grad()

            # take gradient of loss
            l.backward()
            optimizer.step()



        # valid predictions
        print("Loading validation data batch for epoch", epoch, "...")
        for local_batch, local_labels in validation_generator:
            valid_pred = model.forward(local_batch)

            acc = accuracy(valid_pred, local_labels)
            valid_logger.add_scalar('Valid Accuracy', acc, global_step=global_step)

        # iterate global_step
        global_step += 1

    save_model(model)

# TODO: ensure you're computing loss and accuracy correctly

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    parser.add_argument('-lr', '--learning_rate', default=0.01)
    parser.add_argument('-bs', '--batch_size', default=128)
    parser.add_argument('-ne', '--n_epochs', default=100)
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
