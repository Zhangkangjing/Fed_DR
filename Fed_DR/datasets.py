import numpy as np
import torch
import matplotlib.image as mpimg
import urllib.request
import argparse
import zipfile
import os
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import random_split
import gdown
parser = argparse.ArgumentParser(description="Flower Embedded devices")
parser.add_argument(
        "--num_samples",
        type=int,
        default=2900, # Total samples (adjust per run)

        help="Number of traindataset of federated learning ",
    )
parser.add_argument(
        "--trainset_1_num",
        type=int,
        default=1000,  # Client 1 samples (adjust per run)

        help="Number of trainset_1_num of federated learning ",
    )
parser.add_argument(
        "--dataset",
        type=str,
        default='dr',
    )
#parser.add_argument("--cid",type=int,required=True,help="Client id. Should be an integer between 0 and NUM_CLIENTS",)


class DRDataset(Dataset):
    def __init__(self, data_label, data_dir, transform):
        super().__init__()
        self.data_label = data_label
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_label)

    def __getitem__(self, index):
        img_name = self.data_label.id_code[index] + '.png'
        label = self.data_label.diagnosis[index]
        img_path = os.path.join(self.data_dir, img_name)
        image = mpimg.imread(img_path)
        image = (image + 1) * 127.5
        image = image.astype(np.uint8)
        image = self.transform(image)
        return image, label


def get_dataset(args):
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        if args.dataset == 'cifar10':
            train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                             transform=apply_transform)
            test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                            transform=apply_transform)
        elif args.dataset == 'cifar100':
            train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                              transform=apply_transform)
            test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                             transform=apply_transform)

    elif args.dataset == 'mnist' or args.dataset == 'fmnist':
        if args.dataset == 'mnist':
            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))])

            data_dir = '../data/mnist/'
            train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                           transform=apply_transform)
            test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                          transform=apply_transform)

        else:
            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))])

            data_dir = '../data/fmnist/'
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                                  transform=apply_transform)
            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                                 transform=apply_transform)

    elif args.dataset == 'dr':
            data_dir = './data/'
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)

            test_csv_path = data_dir + 'test_set.csv'
            train_csv_path = data_dir + 'train_set.csv'

            df_train = pd.read_csv(train_csv_path)
            df_test = pd.read_csv(test_csv_path)

            # create train and test datasets
            apply_transform = transforms.Compose([transforms.ToPILImage(mode='RGB'),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.Resize(265),
                                                  transforms.CenterCrop(224),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

            image_directory = data_dir + 'images/'
            train_dataset = DRDataset(data_label=df_train, data_dir=image_directory,
                                      transform=apply_transform)
            test_dataset = DRDataset(data_label=df_test, data_dir=image_directory,
                                     transform=apply_transform)

    return train_dataset, test_dataset




def prepare_dataset(num_samples,trainset_1_num, batch_size:int, val_ratio: float = 0.1):
    args = parser.parse_args()
    train_dataset,test_dataset = get_dataset(args)
    # Randomly sample num_samples data points
    train_dataset_subset, _ = random_split(train_dataset, [num_samples, len(train_dataset) - num_samples], torch.Generator().manual_seed(2023))
    trainset_1_num = trainset_1_num
    trainset_2_num = len(train_dataset_subset) - trainset_1_num
    partition_len = [trainset_1_num, trainset_2_num]
    trainsets = random_split(train_dataset_subset, partition_len, torch.Generator().manual_seed(2023))
    trainloaders = []
    valloaders = []

    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val
        for_train, for_val = random_split(trainset_, [num_train, num_val], torch.Generator().manual_seed(2023))
        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(for_val, batch_size=64, shuffle=False, num_workers=2))
    testloader = DataLoader(test_dataset, batch_size=64)
    return trainloaders, valloaders, testloader



## For test
if __name__ == '__main__':
    num_samples = 1300
    trainset_1_num = 500
    trainloaders, valloaders,testloader  = prepare_dataset(num_samples,trainset_1_num, batch_size=32)