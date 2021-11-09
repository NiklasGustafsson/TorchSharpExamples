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
    /// Simple MNIST Convolutional model.
    /// </summary>
    /// <remarks>
    /// There are at least two interesting data sets to use with this example:
    /// 
    /// 1. The classic MNIST set of 60000 images of handwritten digits.
    ///
    ///     It is available at: http://yann.lecun.com/exdb/mnist/
    ///     
    /// 2. The 'fashion-mnist' data set, which has the exact same file names and format as MNIST, but is a harder
    ///    data set to train on. It's just as large as MNIST, and has the same 60/10 split of training and test
    ///    data.
    ///    It is available at: https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion
    ///
    /// In each case, there are four .gz files to download. Place them in a folder and then point the '_dataLocation'
    /// constant below at the folder location.
    /// </remarks>
    public class MNIST
    {
        private static int _epochs = 4;
        private static int _trainBatchSize = 64;

        private readonly static int miniBatchMultiplier = 100;

        internal static void Run(int epochs, string benchmarks, int batchsize)
        {
            _epochs = epochs;
            _trainBatchSize = batchsize;

            var device = cuda.is_available() ? CUDA : CPU;

            var data = Enumerable.Range(0, miniBatchMultiplier).Select(i => torch.rand(_trainBatchSize, 1, 28, 28, device: device)).ToList();
            var labels = Enumerable.Range(0, miniBatchMultiplier).Select(i => torch.randint(0, 10, _trainBatchSize, dtype: int64, device: device)).ToList();

            if (benchmarks.Contains("FWDU")) ForwardPassWithUsings(data, device);
            if (benchmarks.Contains("FWDNU")) ForwardPassNoUsings(data, device);

            if (benchmarks.Contains("BPNU")) BackpropNoUsings(data, labels, device);
            if (benchmarks.Contains("BPU")) BackpropWithUsings(data, labels, device);
        }

        private static void ForwardPassWithUsings(IList<Tensor> data, Device device)
        {
            Console.WriteLine();
            Console.WriteLine($"\tRunning MNIST ForwardPassWithUsings with random data on {device.type.ToString()} for {_epochs}*{miniBatchMultiplier} iterations with a batch size of {_trainBatchSize}.");

            var model = new TorchSharp.Examples.MNIST.Model(device);

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

        private static void ForwardPassNoUsings(IList<Tensor> data, Device device)
        {
            Console.WriteLine();
            Console.WriteLine($"\tRunning MNIST ForwardPassNoUsings with random data on {device.type.ToString()} for {_epochs}*{miniBatchMultiplier} iterations with a batch size of {_trainBatchSize}.");

            var model = new TorchSharp.Examples.MNIST.Model(device);

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
                    var output = model.forwardNoUsings(data[batch]);
                    GC.Collect();
                }
            }

            totalTime.Stop();
            Console.WriteLine($"\tElapsed time: {totalTime.Elapsed.TotalSeconds:F1} s.");
        }

        private static void BackpropWithUsings(IList<Tensor> data, IList<Tensor> labels, Device device)
        {
            Console.WriteLine();
            Console.WriteLine($"\tRunning MNIST BackpropWithUsings with random data on {device.type.ToString()} for {_epochs}*{miniBatchMultiplier} iterations with a batch size of {_trainBatchSize}.");

            var model = new TorchSharp.Examples.MNIST.Model(device);

            // Warm-up

            using (var output = model.forward(data[0])) { }

            model.Train();

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
                    using (var output = loss(prediction, labels[batch]))
                    {
                        output.backward();
                        optimizer.step();
                    }
                }
            }

            totalTime.Stop();
            Console.WriteLine($"\tElapsed time: {totalTime.Elapsed.TotalSeconds:F1} s.");
        }

        private static void BackpropNoUsings(IList<Tensor> data, IList<Tensor> labels, Device device)
        {
            Console.WriteLine();
            Console.WriteLine($"\tRunning MNIST BackpropNoUsings with random data on {device.type.ToString()} for {_epochs}*{miniBatchMultiplier} iterations with a batch size of {_trainBatchSize}.");
            Console.WriteLine($"\tCreating the model...");

            var model = new TorchSharp.Examples.MNIST.Model(device);

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
                    var output = loss(prediction, labels[batch]);

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
