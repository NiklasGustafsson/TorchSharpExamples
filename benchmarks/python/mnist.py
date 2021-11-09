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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.seq = nn.Sequential(
          nn.Conv2d(1, 32, 3, 1),
          nn.ReLU(),
          nn.Conv2d(32, 64, 3, 1),
          nn.ReLU(),
          nn.MaxPool2d(2),
          nn.Dropout(0.25),
          nn.Flatten(),
          nn.Linear(9216, 128),
          nn.ReLU(),
          nn.Dropout(0.5),
          nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.seq(x)
        output = F.log_softmax(x, dim=1)
        return output


def forwardPassOnly(model, epochs, data, device):

    print("\tRunning MNIST forwardPassOnly with random data for", epochs,"*", len(data), "iterations with a batch size of", data[0].shape[0], " running on", device)

    # Warm-up

    _ = model(data[0])

    start_time = time.time()

    for i in range(epochs):
        for d in data:
            
            model(d)

    elapsed = time.time() - start_time

    print('Elapsed time: {:5.2f}s'.format(elapsed))

def backpropagation(model, epochs, data, labels, device):

    print("\tRunning MNIST forwardPassOnly with random data for", epochs,"*", len(data), "iterations with a batch size of", data[0].shape[0], " running on", device)

    # Warm-up

    _ = model(data[0])

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    
    start_time = time.time()

    for i in range(epochs):
        for di in range(len(data)):

            optimizer.zero_grad()

            prediction = model(data[di])
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

    data = [torch.rand(args.batch_size, 1, 28, 28).to(device) for _ in range(args.batches)]
    labels = [torch.randint(0, 10, (args.batch_size,), dtype=torch.int64).to(device) for _ in range(args.batches)]

    model = Net().to(device)

    benchmarks = args.benchmark.split('|')

    for b in benchmarks:
        if b == "FWD": 
            forwardPassOnly(model, args.epochs, data, device)
        elif b == "BPRP":
            backpropagation(model, args.epochs, data, labels, device)
        

if __name__ == '__main__':
    main()
