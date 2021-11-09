from __future__ import print_function
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

def forwardPassOnly(model, modelName, epochs, data, device):

    print("\tRunning", modelName, "forwardPassOnly with random data for", epochs,"*", len(data), "iterations with a batch size of", data[0].shape[0], " running on", device)

    # Warm-up

    _ = model(data[0])

    start_time = time.time()

    for i in range(epochs):
        for d in data:
            
            model(d)

    elapsed = time.time() - start_time

    print('Elapsed time: {:5.2f}s'.format(elapsed))

def backpropagation(model, modelName, epochs, data, labels, device):

    print("\tRunning", modelName, "backpropagation with random data for", epochs,"*", len(data), "iterations with a batch size of", data[0].shape[0], " running on", device)

    # Warm-up

    _ = model(data[0])

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    
    start_time = time.time()

    for i in range(epochs):
        for di in range(len(data)):

            optimizer.zero_grad()

            prediction = F.log_softmax(model(data[di]), 1)
            output = F.nll_loss(prediction, labels[di])

            output.backward()

            optimizer.step()

    elapsed = time.time() - start_time

    print('Elapsed time: {:5.2f}s'.format(elapsed))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--model', type=str, default="Resnet18", metavar='N',
                        help='the number of simulated mini-batched (default: 100)')
    parser.add_argument('--batches', type=int, default=100, metavar='N',
                        help='the number of simulated mini-batched (default: 100)')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--benchmark', type=str, default="FWD|BPRP",
                        help='disables CUDA training')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)

    data = [torch.rand(args.batch_size, 3, 32, 32).to(device) for _ in range(args.batches)]
    labels = [torch.randint(0, 10, (args.batch_size,), dtype=torch.int64).to(device) for _ in range(args.batches)]

    if args.model == "ResNet18":
        model = ResNet18().to(device)
    elif args.model == "ResNet34":
        model = ResNet34().to(device)
    else:
        model = ResNet50().to(device)

    benchmarks = args.benchmark.split('|')

    for b in benchmarks:
        if b == "FWD": 
            forwardPassOnly(model, args.model, args.epochs, data, device)
        elif b == "BPRP":
            backpropagation(model, args.model, args.epochs, data, labels, device)

if __name__ == '__main__':
    main()