import pystk
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from tournament import dense_transforms

DATASET_PATH = 'human_data'

class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor()):

        from PIL import Image
        from glob import glob
        from os import path
        self.data = []
        for f in glob(path.join(dataset_path, '*_ball.csv')):
            i = Image.open(f.replace('_ball.csv', '.png'))
            i.load()
            self.data.append(
                (i, np.loadtxt(f, dtype=np.float32, delimiter=',')))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = self.transform(*data)
        return data


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path, transform=transforms.ToTensor()):
        """
        Your code here
        Hint: Use your solution (or the master solution) to HW1 / HW2
        Hint: If you're loading (and storing) PIL images here, make sure to call image.load(),
              to avoid an OS error for too many open files.
        """
        import csv
        from os import path
        self.data = []
        self.transform = transform
        with open(path.join(dataset_path, 'labels.csv'), newline='') as f:
            reader = csv.reader(f)
            for fname, label, _ in reader:
                if label in LABEL_NAMES:
                    image = Image.open(path.join(dataset_path, fname))
                    image.load()
                    label_id = LABEL_NAMES.index(label)
                    self.data.append((image, label_id))

    def __len__(self):
        """
        Your code here
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Your code here
        Hint: Make sure to apply the transform here, if you use any randomized transforms.
              This ensures that a different random transform is used every time
        """
        # return self.data[idx]
        img, lbl = self.data[idx]
        return self.transform(img), lbl


def load_data(dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor(), num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path, transform=transform)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)
