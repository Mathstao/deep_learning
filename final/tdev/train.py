from .planner import Planner, save_model, load_model
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms

def train(args):
    from os import path
    model = Planner()
    #model = load_model()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))

    """
    Your code here, modify your HW4 code
    
    """
    # Data to device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #device = torch.device('cpu')
    print('device = ', device)

    model.to(device)

    #pos_weight
    pw = args.pos_weight
    if args.pos_weight is not None:
        pw = float(args.pos_weight)
        pw = torch.FloatTensor([pw,pw])
        pw = torch.reshape(pw,(1,2)).to(device)

    # Create loss
    #loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    #loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_fn = torch.nn.MSELoss(reduction='none').to(device)

    # Create loss
    #loss_fn = torch.nn.CrossEntropyLoss()
    #loss_fn.to(device)

    # Transforms
    #img_transforms = dense_transforms.Compose([dense_transforms.RandomHorizontalFlip(),
    #    dense_transforms.ColorJitter(0,4,0.4,0.4),
    #    dense_transforms.ToTensor()])
    img_transforms = dense_transforms.ToTensor()

    # Load data: train and valid
    #input = data/train
    #target = data/valid
    train_loader = load_data(dataset_path=args.data, batch_size=int(args.batch_size), transform=img_transforms)

    # Run SGD for severall epochs
    n_epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    learning_rate = float(args.learning_rate)

    # Create optimizer
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    global_step = 0

    train_loss = 0
    train_accuracy = 0
    valid_loss = 0
    valid_accuracy = 0

    print('Bout to start', n_epochs, 'epochs')

    for step in range(n_epochs):
        # In train mode
        model.train()
        train_batch_loss = []
        train_batch_acc = []
        for im, xy in train_loader:
            im, xy =  im.to(device), xy.to(device)

            h, w = im.size()[2], im.size()[3]

            #modifying hm to help with model from noise
            #if global_step < 10:
            #    hm = hm*100

            #add im and xy to tensorboard to verify

            #print('im',im,'im shape',im.shape)
            #print('xy',xy,'im shape',xy.shape)

            #norm_x = (xy[0]/63.5)-1
            #norm_y = (xy[1]/47.5)-1
            #norm_xy = ((xy[0]/64)-1,(xy[1]/48)-1)

            #m_factor = torch.Tensor([(1/64,0.),(0.,1/48)]).to(device)
            #xy = xy.mm(m_factor)+(-1)
            #xy.to(device)

            #print('new xy',xy,'im shape',xy.shape)

            output_pred = model(im)

            #print('output_pred',output_pred,'output_pred shape',output_pred.shape)

            ### One hot images of xy and output_pred
            #onehot_xy = torch.zeros(batch_size,w,h,requires_grad=True).float().to(device)
            #for idx, pt in enumerate(xy):
            #    onehot_xy[idx][int(pt[0])][int(pt[1])] = 2

            #onehot_output_pred = torch.zeros(batch_size,w,h,requires_grad=True).float().to(device)
            #for idx, pt in enumerate(output_pred):
            #    onehot_output_pred[idx][int(pt[0])][int(pt[1])] = 2

            output_pred = output_pred.to(device)
            xy = xy.to(device)

            # xy scaling x/63.5 - 1

            #scaling_factor = torch.FloatTensor([[1/63.5,0.],[0.,1/47.5]]).to(device)
            scaling_factor = torch.FloatTensor([[1/((w-1)/2),0.],[0.,1/((h-1)/2)]]).to(device)
            xy_val = xy.mm(scaling_factor)
            xy_val = xy_val-1.0

            out_xy = output_pred.mm(scaling_factor)
            out_xy = out_xy-1.0

            xy_val = (xy_val*2)
            out_xy = (out_xy*2)

            xy_val = xy_val.to(device)
            out_xy = out_xy.to(device)

            loss = loss_fn(output_pred, xy)

            #print('loss size',len(loss))

            #loss x val
            loss = loss.chunk(2,dim=1)[0]
            loss = loss.mean()

            if global_step%100==0:
                #print('out pred',output_pred,'xy',xy)
                print('loss',loss)

            train_batch_loss.append(loss.detach().cpu().numpy())

            #print('Loss',loss)
            if args.log_dir is not None:
                train_logger.add_scalar('loss', loss.float(), global_step=global_step)

            # add image to tensorboard
            #train_logger.add_image('im', im, global_step=global_step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

        train_loss = np.mean(train_batch_loss)

        #print("Training loss",train_loss, "Valid loss", valid_loss)
        print('epoch %-3d \t loss = %0.3f' % (step, train_loss))

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--epochs', default=5)
    parser.add_argument('--learning_rate', default=1e-3)
    parser.add_argument('--pos_weight', default=None)
    parser.add_argument('--data', default='drive_data')

    args = parser.parse_args()
    train(args)
