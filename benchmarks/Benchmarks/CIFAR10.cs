// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;

using TorchSharp;
using TorchSharp.torchvision;

using TorchSharp.Examples;
using TorchSharp.Examples.Utils;

using static TorchSharp.torch;

using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace CSharpExamples
{
    /// <summary>
    /// Driver for various models trained and evaluated on the CIFAR10 small (32x32) color image data set.
    /// </summary>
    /// <remarks>
    /// The dataset for this example can be found at: https://www.cs.toronto.edu/~kriz/cifar.html
    /// Download the binary file, and place it in a dedicated folder, e.g. 'CIFAR10,' then edit
    /// the '_dataLocation' definition below to point at the right folder.
    ///
    /// Note: so far, CIFAR10 is supported, but not CIFAR100.
    /// </remarks>
    class CIFAR10
    {
        private static int _epochs = 4;
        private static int _trainBatchSize = 64;

        private readonly static int miniBatchMultiplier = 100;
        private readonly static int _numClasses = 10;

        internal static void Run(int epochs, string benchmarks, int batchsize, string modelName)
        {
            _epochs = epochs;
            _trainBatchSize = batchsize;

            var device =
                // This worked on a GeForce RTX 2080 SUPER with 8GB, for all the available network architectures.
                // It may not fit with less memory than that, but it's worth modifying the batch size to fit in memory.
                torch.cuda.is_available() ? torch.CUDA :
                torch.CPU;

            TorchSharp.Examples.SpecialModule model = null;

            switch (modelName.ToLower()) {
            case "alexnet":
                model = new AlexNet("AlexNet", _numClasses, device);
                break;
            case "mobilenet":
                model = new MobileNet("MobileNet", _numClasses, device);
                break;
            case "vgg11":
            case "vgg13":
            case "vgg16":
            case "vgg19":
                model = new VGG(modelName.ToUpper(), _numClasses, device);
                break;
            case "resnet18":
                model = ResNet.ResNet18(_numClasses, device);
                break;
            case "resnet34":
                model = ResNet.ResNet34(_numClasses, device);
                break;
            case "resnet50":
                model = ResNet.ResNet50(_numClasses, device);
                break;
            case "resnet101":
                model = ResNet.ResNet101(_numClasses, device);
                break;
            case "resnet152":
                model = ResNet.ResNet152(_numClasses, device);
                break;
            }

            var data = Enumerable.Range(0,miniBatchMultiplier).Select(i => torch.rand(_trainBatchSize, 3, 32, 32, device: device)).ToList();
            var labels = Enumerable.Range(0, miniBatchMultiplier).Select(i => torch.randint(0, _numClasses, _trainBatchSize, dtype: int64, device: device)).ToList();

            if (benchmarks.Contains("FWDU")) ForwardPassWithUsings(model, data, modelName, device);
            if (benchmarks.Contains("FWDNU")) ForwardPassNoUsings(model, data, modelName, device);

            if (benchmarks.Contains("BPNU")) BackpropNoUsings(model, data, labels, modelName, device);
            if (benchmarks.Contains("BPU")) BackpropWithUsings(model, data, labels, modelName, device);

            model.Dispose();
        }

        private static void ForwardPassWithUsings(TorchSharp.Examples.SpecialModule model, IList<Tensor> data, string modelName, Device device)
        {
            Console.WriteLine();
            Console.WriteLine($"\tRunning {modelName} ForwardPassWithUsings with random data on {device.type.ToString()} for {_epochs}*{data.Count()} iterations with a batch size of {_trainBatchSize}.");

            // Warm-up

            using (var output = model.forward(data[0])) { }

            Console.WriteLine($"\tStart timing...");
            Stopwatch totalTime = new Stopwatch();
            totalTime.Start();

            for (int i = 0; i < _epochs; i++)
            {
                for (var batch = 0; batch < data.Count(); batch++)
                {
                    using (var output = model.forward(data[batch])) { }
                }
            }

            totalTime.Stop();
            Console.WriteLine($"\tElapsed time: {totalTime.Elapsed.TotalSeconds:F1} s.");
        }

        private static void ForwardPassNoUsings(TorchSharp.Examples.SpecialModule model, IList<Tensor> data, string modelName, Device device)
        {
            Console.WriteLine();
            Console.WriteLine($"\tRunning {modelName} ForwardPassNoUsings with random data on {device.type.ToString()} for {_epochs}*{data.Count()} iterations with a batch size of {_trainBatchSize}.");

            // Warm-up

            using (var output = model.forwardNoUsings(data[0])) { }
            GC.Collect();

            Console.WriteLine($"\tStart timing...");
            Stopwatch totalTime = new Stopwatch();
            totalTime.Start();

            for (int i = 0; i < _epochs; i++)
            {
                for (var batch = 0; batch < data.Count(); batch++)
                {
                    using (var output = model.forwardNoUsings(data[batch])) { }
                    GC.Collect();
                }
            }

            totalTime.Stop();
            Console.WriteLine($"\tElapsed time: {totalTime.Elapsed.TotalSeconds:F1} s.");
        }

        private static void BackpropWithUsings(TorchSharp.Examples.SpecialModule model, IList<Tensor> data, IList<Tensor> labels, string modelName, Device device)
        {
            Console.WriteLine();
            Console.WriteLine($"\tRunning {modelName} BackpropWithUsings with random data on {device.type.ToString()} for {_epochs}*{data.Count()} iterations with a batch size of {_trainBatchSize}.");

            // Warm-up

            using (var output = model.forward(data[0])) { }

            var loss = nll_loss(reduction: Reduction.Mean);
            var optimizer = optim.Adam(model.parameters());

            model.Train();

            Console.WriteLine($"\tStart timing...");
            Stopwatch totalTime = new Stopwatch();
            totalTime.Start();

            for (int i = 0; i < _epochs; i++)
            {
                for (var batch = 0; batch < data.Count(); batch++)
                {
                    optimizer.zero_grad();

                    using (var prediction = model.forward(data[batch]))
                    using (var lsm = log_softmax(prediction, 1))
                    using (var output = loss(lsm, labels[batch]))
                    {
                        output.backward();
                        optimizer.step();
                    }
                }
            }

            totalTime.Stop();
            Console.WriteLine($"\tElapsed time: {totalTime.Elapsed.TotalSeconds:F1} s.");
        }

        private static void BackpropNoUsings(TorchSharp.Examples.SpecialModule model, IList<Tensor> data, IList<Tensor> labels, string modelName, Device device)
        {
            Console.WriteLine();
            Console.WriteLine($"\tRunning {modelName} BackpropNoUsings with random data on {device.type.ToString()} for {_epochs}*{data.Count()} iterations with a batch size of {_trainBatchSize}.");
            Console.WriteLine($"\tCreating the model...");

            // Warm-up

            using (var output = model.forwardNoUsings(data[0])) { }

            var loss = nll_loss(reduction: Reduction.Mean);
            var optimizer = optim.Adam(model.parameters());

            model.Train();

            Console.WriteLine($"\tStart timing...");
            Stopwatch totalTime = new Stopwatch();
            totalTime.Start();

            for (int i = 0; i < _epochs; i++)
            {
                for (var batch = 0; batch < data.Count(); batch++)
                {
                    optimizer.zero_grad();

                    var prediction = model.forwardNoUsings(data[batch]);
                    var lsm = log_softmax(prediction, 1);
                    var output = loss(lsm, labels[batch]);

                    output.backward();
                    optimizer.step();

                    GC.Collect();
                }
            }

            totalTime.Stop();
            Console.WriteLine($"\tElapsed time: {totalTime.Elapsed.TotalSeconds:F1} s.");
        }
    }
}
