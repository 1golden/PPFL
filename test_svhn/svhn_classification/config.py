import os
import torchvision
import flgo.benchmark
import torch

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
root = os.path.join(flgo.benchmark.path, 'RAW_DATA', 'SVHN')  # 可以为任意存放原始数据的绝对路径
train_data = torchvision.datasets.SVHN(root=root, transform=transform, download=True, split='train')
test_data = torchvision.datasets.SVHN(root=root, transform=transform, download=True, split='test')


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.embedder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 64, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(1),
            torch.nn.Linear(1600, 384),
            torch.nn.ReLU(),
            torch.nn.Linear(384, 192),
            torch.nn.ReLU(),
        )
        self.fc = torch.nn.Linear(192, 10)

    def forward(self, x):
        x = self.embedder(x)
        return self.fc(x)


def get_model():
    return CNN()
