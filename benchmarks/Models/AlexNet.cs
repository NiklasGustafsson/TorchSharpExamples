// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;

using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace TorchSharp.Examples
{
    public abstract class SpecialModule : Module
    {
        public SpecialModule(string name) : base(name) { }

        public abstract Tensor forwardNoUsings(Tensor input);

    }

    /// <summary>
    /// Modified version of original AlexNet to fix CIFAR10 32x32 images.
    /// </summary>
    public class AlexNet : SpecialModule
    {
        private readonly Modules.ModuleList features;
        private readonly Modules.ModuleList classifier;

        public AlexNet(string name, int numClasses, Device device = null) : base(name)
        {
            features = ModuleList(
                Conv2d(3, 64, kernelSize: 3, stride: 2, padding: 1),
                ReLU(inPlace: true),
                MaxPool2d(kernelSize: new long[] { 2, 2 }),
                Conv2d(64, 192, kernelSize: 3, padding: 1),
                ReLU(inPlace: true),
                MaxPool2d(kernelSize: new long[] { 2, 2 }),
                Conv2d(192, 384, kernelSize: 3, padding: 1),
                ReLU(inPlace: true),
                Conv2d(384, 256, kernelSize: 3, padding: 1),
                ReLU(inPlace: true),
                Conv2d(256, 256, kernelSize: 3, padding: 1),
                ReLU(inPlace: true),
                MaxPool2d(kernelSize: new long[] { 2, 2 }),
                AdaptiveAvgPool2d(new long[] { 2, 2 }));
            classifier = ModuleList(
                Dropout(),
                Linear(256 * 2 * 2, 4096),
                ReLU(inPlace: true),
                Dropout(),
                Linear(4096, 4096),
                ReLU(inPlace: true),
                Dropout(),
                Linear(4096, numClasses));

            RegisterComponents();

            if (device != null && device.type == DeviceType.CUDA)
                this.to(device);
        }

        public override Tensor forward(Tensor input)
        {
            var t0 = features[0].forward(input);

            for (var i = 1; i < features.Count; i++)
            {
                var t1 = features[i].forward(t0);
                t0.Dispose();
                t0 = t1;
            }

            var x = t0.reshape(new long[] { t0.shape[0], 256 * 2 * 2 });

            t0 = classifier[0].forward(x);
            x.Dispose();

            for (var i = 0; i < classifier.Count-1; i++)
            {
                var t1 = classifier[i].forward(t0);
                t0.Dispose();
                t0 = t1;
            }

            var result = classifier[classifier.Count - 1].forward(t0);
            t0.Dispose();
            return result;
        }

        public override Tensor forwardNoUsings(Tensor input)
        {
            Tensor result = input;

            for (var i = 0; i < features.Count; i++)
            {
                result = features[i].forward(result);
            }

            result = result.reshape(new long[] { result.shape[0], 256 * 2 * 2 });

            for (var i = 0; i < classifier.Count; i++)
            {
                result = classifier[i].forward(result);
            }

            return result;
        }
    }

}
