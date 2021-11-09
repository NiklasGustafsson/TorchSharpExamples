// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System.Collections.Generic;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchSharp.Examples
{
    /// <summary>
    /// Modified version of VGG to classify CIFAR10 32x32 images.
    /// </summary>
    /// <remarks>
    /// With an unaugmented CIFAR-10 data set, the author of this saw training converge
    /// at roughly 85% accuracy on the test set, after 50 epochs using VGG-16.
    /// </remarks>
    public class VGG : SpecialModule
    {
        // The code here is is loosely based on https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py
        // Licence and copypright notice at: https://github.com/kuangliu/pytorch-cifar/blob/master/LICENSE

        private readonly Dictionary<string, long[]> _channels = new Dictionary<string, long[]>() {
            { "vgg11", new long[] { 64, 0, 128, 0, 256, 256, 0, 512, 512, 0, 512, 512, 0 } },
            { "vgg13", new long[] { 64, 64, 0, 128, 128, 0, 256, 256, 0, 512, 512, 0, 512, 512, 0 } },
            { "vgg16", new long[] { 64, 64, 0, 128, 128, 0, 256, 256, 256, 0, 512, 512, 512, 0, 512, 512, 512, 0 } },
            { "vgg19", new long[] { 64, 64, 0, 128, 128, 0, 256, 256, 256, 256, 0, 512, 512, 512, 512, 0, 512, 512, 512, 512, 0 } }
        };

        private readonly Modules.ModuleList layers;

        public VGG(string name, int numClasses, Device device = null) : base(name)
        {
            var modules = new List<Module>();

            var channels = _channels[name.ToLower()];

            long in_channels = 3;

            for (var i = 0; i < channels.Length; i++) {

                if (channels[i] == 0) {
                    modules.Add(MaxPool2d(kernelSize: 2, stride: 2));
                } else {
                    modules.Add(Conv2d(in_channels, channels[i], kernelSize: 3, padding: 1));
                    modules.Add(BatchNorm2d(channels[i]));
                    modules.Add(ReLU(inPlace: true));
                    in_channels = channels[i];
                }
            }
            modules.Add(AvgPool2d(kernel_size: 1, stride: 1));
            modules.Add(Flatten());
            modules.Add(Linear(512, numClasses));

            layers = ModuleList(modules.ToArray());

            RegisterComponents();

            if (device != null && device.type == DeviceType.CUDA)
                this.to(device);
        }

        public override Tensor forward(Tensor input)
        {
            var t0 = layers[0].forward(input);

            for (var i = 1; i < layers.Count-1; i++)
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
