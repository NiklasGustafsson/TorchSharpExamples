// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;

using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchSharp.Examples
{
    /// <summary>
    /// Modified version of MobileNet to classify CIFAR10 32x32 images.
    /// </summary>
    /// <remarks>
    /// With an unaugmented CIFAR-10 data set, the author of this saw training converge
    /// at roughly 75% accuracy on the test set, over the course of 1500 epochs.
    /// </remarks>
    public class MobileNet : SpecialModule
    {
        // The code here is is loosely based on https://github.com/kuangliu/pytorch-cifar/blob/master/models/mobilenet.py
        // Licence and copypright notice at: https://github.com/kuangliu/pytorch-cifar/blob/master/LICENSE

        private readonly long[] planes = new long[] { 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024 };
        private readonly long[] strides = new long[] { 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1 };

        private readonly Modules.ModuleList layers;

        public MobileNet(string name, int numClasses, Device device = null) : base(name)
        {
            if (planes.Length != strides.Length) throw new ArgumentException("'planes' and 'strides' must have the same length.");

            var modules = new List<Module>();

            modules.Add(Conv2d(3, 32, kernelSize: 3, stride: 1, padding: 1, bias: false));
            modules.Add(BatchNorm2d(32));
            modules.Add(ReLU());
            MakeLayers(modules, 32);
            modules.Add(AvgPool2d(new long[] { 2, 2 }));
            modules.Add(Flatten());
            modules.Add(Linear(planes[^1], numClasses));

            layers = ModuleList(modules.ToArray());

            RegisterComponents();

            if (device != null && device.type == DeviceType.CUDA)
                this.to(device);
        }

        private void MakeLayers(List<Module> modules, long in_planes)
        {

            for (var i = 0; i < strides.Length; i++) {

                var out_planes = planes[i];
                var stride = strides[i];

                modules.Add(Conv2d(in_planes, in_planes, kernelSize: 3, stride: stride, padding: 1, groups: in_planes, bias: false));
                modules.Add(BatchNorm2d(in_planes));
                modules.Add(ReLU());
                modules.Add(Conv2d(in_planes, out_planes, kernelSize: 1L, stride: 1L, padding: 0L, bias: false));
                modules.Add(BatchNorm2d(out_planes));
                modules.Add(ReLU());

                in_planes = out_planes;
            }
        }

        public override Tensor forward(Tensor input)
        {
            var t0 = layers[0].forward(input);

            for (var i = 1; i < layers.Count - 1; i++)
            {
                var t1 = layers[i].forward(t0);
                t0.Dispose();
                t0 = t1;
            }

            var result = layers[layers.Count - 1].forward(t0);
            t0.Dispose();
            return result;
        }

        public override Tensor forwardNoUsings(Tensor input)
        {
            Tensor result = input;

            for (var i = 0; i < layers.Count; i++)
            {
                result = layers[i].forward(result);
            }

            return result;
        }
    }
}
