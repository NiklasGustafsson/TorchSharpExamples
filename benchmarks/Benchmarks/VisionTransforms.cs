// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;

using TorchSharp;
using TorchSharp.torchvision;

using static TorchSharp.torch;

using TorchSharp.Examples;
using TorchSharp.Examples.Utils;

namespace CSharpExamples
{
    class VisionTransforms
    {
        private static int _epochs = 4;
        private static int _trainBatchSize = 64;

        private readonly static int miniBatchMultiplier = 50;

        internal static void Run(int epochs, int batches, int batchsize, string modelName)
        {
            _epochs = epochs;
            _trainBatchSize = batchsize;

            if (batches == -1) batches = miniBatchMultiplier;

            var device =
                // This worked on a GeForce RTX 2080 SUPER with 8GB, for all the available network architectures.
                // It may not fit with less memory than that, but it's worth modifying the batch size to fit in memory.
                torch.cuda.is_available() ? torch.CUDA :
                torch.CPU;

            ITransform model = null;

            switch (modelName.ToLower())
            {
                case "adjusthue":
                    model = transforms.AdjustHue(0.15);
                    break;
                case "rotate":
                    model = transforms.Rotate(90);
                    break;
            }

            var data = Enumerable.Range(0, batches).Select(i => torch.rand(_trainBatchSize, 3, 256, 256, device: device)).ToList();

            transform(model, data, modelName, device);

        }

        private static void transform(ITransform model, IList<Tensor> data, string modelName, Device device)
        {
            Console.WriteLine();
            Console.WriteLine($"\tRunning {modelName} transform() with random data on {device.type.ToString()} for {_epochs}*{data.Count} iterations with a batch size of {_trainBatchSize}.");

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
                    GC.Collect();
                }
            }

            totalTime.Stop();
            Console.WriteLine($"\tElapsed time: {totalTime.Elapsed.TotalSeconds:F1} s.");
        }
    }
}
