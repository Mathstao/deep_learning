from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import csv
import torch

"""
__init__
__len___
__getitem___
The __len__ function should return the size of the dataset.
The __getitem__ function should return a tuple of image, label. The image should be a torch.Tensor of size (3,64,64) with range [0,1], and the label should be int.
Labels and the corresponding image paths are saved in labels.csv, their headers are file and label. There are 6 classes of objects. Make sure label background corresponds to 0, kart is 1, pickup is 2, nitro is 3, bomb is 4 and projectile 5.

"""

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']
label_dict = {'background': 0, 'kart': 1, 'pickup': 2, 'nitro': 3, 'bomb': 4, 'projectile': 5}

class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):
        self.images = []
        self.labels = []
        # setup pytorch transformation
        transformation = transforms.Compose([transforms.ToTensor()])
        # parse csv with csv package, load image with Pillow
        csv_path = str(dataset_path) + "/" + "labels.csv"
        with open(csv_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_num = 0
            for row in csv_reader:
                # first line has data headers
                if line_num > 0:
                    # get image data
                    filename = row[0]
                    img_path = str(dataset_path) + "/" + filename
                    img = Image.open(str(img_path))
                    img_data = transformation(img)
                    # get label
                    label_name = row[1]
                    label = label_dict[label_name]
                    # store to data array of tuples (img_data, label)
                    self.images.append(img_data)
                    self.labels.append(label)
                line_num += 1

    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        return (self.images[idx], self.labels[idx])


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
