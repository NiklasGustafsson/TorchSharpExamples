from __future__ import print_function
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

# import exportsd

class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1

    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()
        self.planes = [64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]
        self.strides = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1]
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for i in range(len(self.strides)):
            out_planes = self.planes[i]
            stride = self.strides[i]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def forwardPassOnly(model, epochs, data, device):

    print("\tRunning MobileNet forwardPassOnly with random data for", epochs,"*", len(data), "iterations with a batch size of", data[0].shape[0], " running on", device)

    # Warm-up

    _ = model(data[0])

    start_time = time.time()

    for i in range(epochs):
        for d in data:
            
            model(d)

    elapsed = time.time() - start_time

    print('Elapsed time: {:5.2f}s'.format(elapsed))

def backpropagation(model, epochs, data, labels, device):

    print("\tRunning MobileNet backpropagation with random data for", epochs,"*", len(data), "iterations with a batch size of", data[0].shape[0], " running on", device)

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

    model = MobileNet().to(device)

    benchmarks = args.benchmark.split('|')

    for b in benchmarks:
        if b == "FWD": 
            forwardPassOnly(model, args.epochs, data, device)
        elif b == "BPRP":
            backpropagation(model, args.epochs, data, labels, device)

if __name__ == '__main__':
    main()
