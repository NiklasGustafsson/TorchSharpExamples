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

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

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
    parser.add_argument('--model', type=str, default="VGG11", metavar='N',
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

    model = VGG(args.model).to(device)

    benchmarks = args.benchmark.split('|')

    for b in benchmarks:
        if b == "FWD": 
            forwardPassOnly(model, args.model, args.epochs, data, device)
        elif b == "BPRP":
            backpropagation(model, args.model, args.epochs, data, labels, device)

if __name__ == '__main__':
    main()
