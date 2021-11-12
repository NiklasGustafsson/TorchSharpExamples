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

def runTransform(transform, epochs, data, modelName, device):

    print("\tRunning", modelName, "forward()) with random data for", epochs,"*", len(data), "iterations with a batch size of", data[0].shape[0], " running on", device)

    # Warm-up

    _ = transform(data[0])

    start_time = time.time()

    for i in range(epochs):
        for d in data:
            x = transform(d)

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

    data = [torch.rand(args.batch_size, 3, 256, 256).to(device) for _ in range(args.batches)]

    benchmarks = args.benchmark.split('|')

    for b in benchmarks:
        if b == "adjusthue": 
            model = transforms.Lambda(lambda data: transforms.functional.adjust_hue(data, 0.15))
            runTransform(model, args.epochs, data, b, device)
        elif b == "rotate": 
            model = transforms.Lambda(lambda data: transforms.functional.rotate(data, 90))
            runTransform(model, args.epochs, data, b, device)
        
if __name__ == '__main__':
    main()
