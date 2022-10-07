from torchvision import datasets, transforms
from data.manipulate import UnNormalize
from avalanche.benchmarks.datasets import core50


# Specify available data-sets
AVAILABLE_DATASETS = {
    'mnist': datasets.MNIST,
    'fmnist': datasets.FashionMNIST,
    'cifar10': datasets.CIFAR10,
    'cifar100': datasets.CIFAR100,
    'core50':  core50.CORe50Dataset
}


# Specify available transforms
AVAILABLE_TRANSFORMS = {
    'mnist': [
        transforms.Pad(2),
        transforms.ToTensor(),
    ],
    'fmnist': [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ],
    'mnist28': [
        transforms.ToTensor(),
    ],
    "core50": [
        transforms.ToTensor(),
    ],
    'cifar10': [
        transforms.ToTensor(),
    ],
    'cifar100': [
        transforms.ToTensor(),
    ],
    'cifar10_norm': [
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ],
    'cifar100_norm': [
        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761])
    ],
    'core50_norm': [
        transforms.Normalize([0.6001, 0.5721, 0.5416], [0.1965, 0.2066, 0.2183])
    ],
    'cifar10_denorm': UnNormalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    'cifar100_denorm': UnNormalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761]),
    'core50_denorm': UnNormalize(mean=[0.6001, 0.5721, 0.5416], std=[0.1965, 0.2066, 0.2183]),
    'augment_from_tensor': [
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4, padding_mode='symmetric'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ],
    'augment': [
        transforms.RandomCrop(32, padding=4, padding_mode='symmetric'),
        transforms.RandomHorizontalFlip(),
    ],
}


# Specify configurations of available data-sets
DATASET_CONFIGS = {
    'mnist': {'size': 32, 'channels': 1, 'classes': 10},
    'fmnist': {'size': 32, 'channels': 1, 'classes': 10},
    'mnist28': {'size': 28, 'channels': 1, 'classes': 10},
    'core50':  {'size': 32, 'channels': 3, 'classes': 50},
    'cifar10': {'size': 32, 'channels': 3, 'classes': 10},
    'cifar100': {'size': 32, 'channels': 3, 'classes': 100},
}