import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import os

def load_cifar10_data(data_dir='./data', batch_size_train=64, batch_size_test=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    os.makedirs(data_dir, exist_ok=True)

    train_data = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size_test, shuffle=False)

    return train_loader, test_loader