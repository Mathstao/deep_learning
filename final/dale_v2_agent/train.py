from .planner import Planner, save_model 
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms

def train(args):
    from os import path
    model = Planner()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))

    """
    Your code here, modify your HW4 code
    
    """
    import torch

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = model.to(device)
    if args.continue_training:
        print('Continue Training')
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'planner.th')))

    loss = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    import inspect
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})

    #train_data = load_data('test_csv_2', transform=transform)
    train_data = load_data('test_csv_2')

    total_mean = torch.rand(1,3).to(device)
    total_std = torch.rand(1,3).to(device)

    global_step = 0
    for epoch in range(args.num_epoch):
        model.train()
        losses = []
        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            h, w = img.size()[2], img.size()[3]

            pred = model(img)   

            #clamping x and y values
            x,y = label.chunk(2, dim=1)
            xy = torch.cat((x.clamp(min=0.0,max=w),y.clamp(min=0.0,max=h)),dim=1)
            xy = xy.to(device)

            loss_val = loss(pred, xy)

            #calc mean and std
            # x (batch, 3, h, w)
            #np.mean(x.numpy(), axis=(0,2,3))
            # array([0.52516913, 0.52009195, 0.48378614]
            # x.mean(dim=(2,3)).mean(dim=0)
            total_mean = torch.cat((total_mean, img.mean(dim=(2,3)).mean(dim=0)[None]))
            total_std = torch.cat((total_std, img.std(dim=(2,3)).std(dim=0)[None]))

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)
                if global_step % 100 == 0:
                    import matplotlib.pyplot as plt
                    import torchvision.transforms.functional as TF
                    fig, ax = plt.subplots(1, 1)
                    ax.imshow(TF.to_pil_image(img[0].cpu()))
                    ax.add_artist(plt.Circle(label[0], 2, ec='g', fill=False, lw=1.5))
                    ax.add_artist(plt.Circle(pred[0], 2, ec='r', fill=False, lw=1.5))
                    train_logger.add_figure('viz', fig, global_step)
                    del ax, fig

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1
            
            losses.append(loss_val.detach().cpu().numpy())
        
        avg_loss = np.mean(losses)
        print('epoch %-3d \t loss = %0.3f' % (epoch, avg_loss))
        save_model(model)

        #print('total_mean shape before mods',total_mean.shape)
        #total_mean = total_mean[1:-1].mean(dim=0)
        #total_std = total_std[1:-1].std(dim=0)
        #print('Epoch total_mean',total_mean[1:-1].mean(dim=0))
        #print('Epoch total_std',total_std[1:-1].std(dim=0))

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform', default='Compose([ColorJitter(0.2, 0.5, 0.5, 0.2), RandomHorizontalFlip(), ToTensor()])')

    args = parser.parse_args()
    train(args)
