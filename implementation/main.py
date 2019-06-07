import torch
import torchvision


class PaddingWrapper(object):
    """
    Wrapper class for torchvision.transforms.functional.pad
    Allows passing the pad function to torchvision.transforms.Compose
    """

    def __init__(self, padding, fill=0, padding_mode='constant'):
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        return torchvision.transforms.functional.pad(img, padding=self.padding, fill=self.fill,
                                                     padding_mode=self.padding_mode)


def loadMNIST(root, download, batch_size, num_workers):
    # pad MNIST 28x28 images with 2 px to reach same size as CIFAR (32x32)
    transform = torchvision.transforms.Compose([
        PaddingWrapper(padding=2, padding_mode='edge'),
        torchvision.transforms.ToTensor()
    ])

    trainset = torchvision.datasets.MNIST(root=root, train=True, download=download, transform=transform)
    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.MNIST(root=root, train=False, download=download, transform=transform)
    testLoader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    return (trainLoader, testLoader, classes)


def loadCIFAR10(root, download, batch_size, num_workers):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=download, transform=transform)
    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root=root, train=False, download=download, transform=transform)
    testLoader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return (trainLoader, testLoader, classes)
