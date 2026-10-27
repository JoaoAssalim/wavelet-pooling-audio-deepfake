import torch
import torch.nn as nn

from deepfake.features import get_node


def conv_block(cin, cout, relu=True):
    layers = [
        nn.Conv1d(cin, cout, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm1d(cout),
    ]
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def make_pooling(pooling, wavelet_family, level, node):
    if pooling == "max":
        return nn.MaxPool1d(kernel_size=2 ** level, ceil_mode=True)
    if pooling == "average":
        return nn.AvgPool1d(kernel_size=2 ** level, ceil_mode=True)
    if pooling == "wavelet":
        return WaveletPooling(wavelet_family, level, node)
    raise ValueError(f"Unknown pooling: {pooling}")


class WaveletPooling(nn.Module):
    def __init__(self, family, level, node):
        super().__init__()
        self.family = family
        self.level = level
        self.node = node

    def forward(self, x):
        return get_node(x, self.family, self.level, self.node)


class CustomCNN(nn.Module):
    def __init__(self, wavelet_family, level, node, pooling):
        super().__init__()

        self.features = nn.Sequential(
            conv_block(1, 16, relu=False),
            conv_block(16, 16, relu=False),
            make_pooling(pooling, wavelet_family, level, node),

            conv_block(16, 64),
            conv_block(64, 64),

            conv_block(64, 128),
            conv_block(128, 128),

            conv_block(128, 256),
            conv_block(256, 256),
            nn.Dropout1d(0.1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.cat([x.mean(dim=2), x.std(dim=2)], dim=1)
        return self.classifier(x)