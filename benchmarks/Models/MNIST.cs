// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.IO;
using System.Collections.Generic;
using System.Diagnostics;
using static TorchSharp.torch;

using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace TorchSharp.Examples.MNIST
{
    public class Model : Module
    {
        private Module conv1 = Conv2d(1, 32, 3);
        private Module conv2 = Conv2d(32, 64, 3);
        private Module fc1 = Linear(9216, 128);
        private Module fc2 = Linear(128, 10);

        // These don't have any parameters, so the only reason to instantiate
        // them is performance, since they will be used over and over.
        private Module pool1 = MaxPool2d(kernelSize: new long[] { 2, 2 });

        private Module relu1 = ReLU();
        private Module relu2 = ReLU();
        private Module relu3 = ReLU();

        private Module dropout1 = Dropout(0.25);
        private Module dropout2 = Dropout(0.5);

        private Module flatten = Flatten();
        private Module logsm = LogSoftmax(1);

        public Model(torch.Device device = null) : base("MNIST")
        {
            RegisterComponents();

            if (device != null && device.type == DeviceType.CUDA)
                this.to(device);
        }

        public override Tensor forward(Tensor input)
        {
            using var l11 = conv1.forward(input);
            using var l12 = relu1.forward(l11);

            using var l21 = conv2.forward(l12);
            using var l22 = relu2.forward(l21);
            using var l23 = pool1.forward(l22);
            using var l24 = dropout1.forward(l23);

            using var x = flatten.forward(l24);

            using var l31 = fc1.forward(x);
            using var l32 = relu3.forward(l31);
            using var l33 = dropout2.forward(l32);

            using var l41 = fc2.forward(l33);

            return logsm.forward(l41);
        }


        public Tensor forwardNoUsings(Tensor input)
        {
            var l12 = relu1.forward(conv1.forward(input));
            var l24 = dropout1.forward(pool1.forward(relu2.forward(conv2.forward(l12))));

            var x = flatten.forward(l24);

            var l33 = dropout2.forward(relu3.forward(fc1.forward(x)));

            return logsm.forward(fc2.forward(l33));
        }
    }
}
