import torch.nn as nn


class ActivationDecoder(nn.Module):
    def __init__(self, in_channels, in_hw, out_hw=64, hidden=256):
        super().__init__()
        layers = []
        c = int(in_channels)
        h = int(in_hw)
        while h < out_hw:
            next_c = max(32, min(hidden, c // 2))
            layers.extend(
                [
                    nn.ConvTranspose2d(c, next_c, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(next_c),
                    nn.ReLU(inplace=True),
                ]
            )
            c = next_c
            h *= 2
        layers.extend([nn.Conv2d(c, 3, kernel_size=3, padding=1), nn.Sigmoid()])
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)

