from collections import OrderedDict
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Sequential):
    def __init__(
        self,
        input_channels: int,
        growth_rate: int,
        bn_size: int,
        drop_p: float = 0.5,
        conv_bias: bool = True,
        batch_norm: bool = True,
    ):
        super(DenseLayer, self).__init__()
        if batch_norm:
            self.add_module("norm1", nn.BatchNorm1d(input_channels)),
        self.add_module("elu1", nn.ELU()),
        self.add_module(
            "conv1",
            nn.Conv1d(
                input_channels,
                bn_size * growth_rate,
                kernel_size=1,
                stride=1,
                bias=conv_bias,
            ),
        ),
        if batch_norm:
            self.add_module("norm2", nn.BatchNorm1d(bn_size * growth_rate)),
        self.add_module("elu2", nn.ELU()),
        self.add_module(
            "conv2",
            nn.Conv1d(
                bn_size * growth_rate,
                growth_rate,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=conv_bias,
            ),
        ),
        self.drop_p = drop_p

    def forward(self, x):
        new_features = super(DenseLayer, self).forward(x)
        new_features = F.dropout(new_features, p=self.drop_p, training=self.training)
        return torch.cat([x, new_features], 1)


class DenseBlock(nn.Sequential):
    def __init__(
        self,
        num_layers,
        input_channels,
        growth_rate,
        bn_size,
        drop_rate=0.5,
        conv_bias=True,
        batch_norm=True,
    ):
        super(DenseBlock, self).__init__()
        for idx_layer in range(num_layers):
            layer = DenseLayer(
                input_channels + idx_layer * growth_rate,
                growth_rate,
                bn_size,
                drop_rate,
                conv_bias,
                batch_norm,
            )
            self.add_module("denselayer%d" % (idx_layer + 1), layer)


class TransitionLayer(nn.Sequential):
    def __init__(
        self, input_channels, output_channels, conv_bias=True, batch_norm=True
    ):
        super(TransitionLayer, self).__init__()
        if batch_norm:
            self.add_module("norm", nn.BatchNorm1d(input_channels))
        self.add_module("elu", nn.ELU())
        self.add_module(
            "conv",
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size=1,
                stride=1,
                bias=conv_bias,
            ),
        )
        self.add_module("pool", nn.AvgPool1d(kernel_size=2, stride=2))


class SPaRCNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        sample_length: int = 2000,
        num_classes: int = 1,
        block_layers=4,
        growth_rate=16,
        bn_size=16,
        drop_rate=0.5,
        conv_bias=True,
        batch_norm=True,
        **kwargs,
    ):
        super(SPaRCNet, self).__init__()

        # add initial convolutional layer
        out_channels = 2 ** (math.floor(np.log2(in_channels)) + 1)
        first_conv = OrderedDict(
            [
                (
                    "conv0",
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=7,
                        stride=2,
                        padding=3,
                        bias=conv_bias,
                    ),
                )
            ]
        )
        first_conv["norm0"] = nn.BatchNorm1d(out_channels)
        first_conv["elu0"] = nn.ELU()
        first_conv["pool0"] = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.encoder = nn.Sequential(first_conv)

        n_channels = out_channels

        # add dense blocks
        for n_layer in np.arange(math.floor(np.log2(sample_length // 4))):
            block = DenseBlock(
                num_layers=block_layers,
                input_channels=n_channels,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                conv_bias=conv_bias,
                batch_norm=batch_norm,
            )
            self.encoder.add_module("denseblock%d" % (n_layer + 1), block)
            # update number of channels after each dense block
            n_channels = n_channels + block_layers * growth_rate

            trans = TransitionLayer(
                input_channels=n_channels,
                output_channels=n_channels // 2,
                conv_bias=conv_bias,
                batch_norm=batch_norm,
            )
            self.encoder.add_module("transition%d" % (n_layer + 1), trans)
            # update number of channels after each transition layer
            n_channels = n_channels // 2

        """ classification layer """
        self.classifier = nn.Sequential(
            nn.ELU(),
            nn.Linear(n_channels, num_classes),
        )

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        emb = self.encoder(x).squeeze(-1)
        out = self.classifier(emb)
        return out


if __name__ == "__main__":
    import time
    X = torch.randn(1, 16, 2560).to('cuda')
    model = SPaRCNet(in_channels=16, sample_length=2560).to('cuda')
    print("Model parameters:", sum(p.numel() for p in model.parameters()))

    t0 = time.time()
    out = model(X)
    t1 = time.time()
    print(f"Inference time: {t1 - t0} seconds")
    print(out.shape)