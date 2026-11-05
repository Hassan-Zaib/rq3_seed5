import torch
import torch.nn as nn


class SpatialLabelMap(nn.Module):
    def __init__(self, num_classes, height, width, embedding_dim=50):
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.project = nn.Linear(embedding_dim, self.height * self.width)

    def forward(self, labels):
        label_map = self.project(self.embedding(labels.long()))
        return label_map.view(labels.shape[0], 1, self.height, self.width)


class ConditionalDecoder(nn.Module):
    def __init__(self, in_channels, in_hw, num_classes=200, out_hw=64):
        super().__init__()
        self.label_map = SpatialLabelMap(num_classes, in_hw, in_hw)
        channels = min(int(in_channels), 256)
        layers = [
            nn.Conv2d(int(in_channels) + 1, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        ]
        spatial = int(in_hw)
        while spatial < out_hw:
            next_channels = max(channels // 2, 64)
            layers.extend(
                [
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(channels, next_channels, 3, padding=1),
                    nn.BatchNorm2d(next_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            channels = next_channels
            spatial *= 2
        layers.extend(
            [
                nn.Conv2d(channels, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 3, 3, padding=1),
                nn.Sigmoid(),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, z, labels):
        return self.net(torch.cat([z, self.label_map(labels)], dim=1))


class ConditionalFeatureDiscriminator(nn.Module):
    def __init__(self, in_channels, in_hw, num_classes=200):
        super().__init__()
        self.label_map = SpatialLabelMap(num_classes, in_hw, in_hw)
        filters = 128 if int(in_channels) <= 128 else 256
        layers = []
        channels = int(in_channels) + 1
        for index, stride in enumerate((1, 2, 2)):
            next_channels = min(filters * (2**index), 512)
            layers.append(nn.Conv2d(channels, next_channels, 3, stride=stride, padding=1))
            if stride != 1:
                layers.append(nn.BatchNorm2d(next_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            channels = next_channels
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(channels * max(1, (int(in_hw) + 3) // 4) ** 2, 1),
        )

    def forward(self, z, labels):
        x = torch.cat([z, self.label_map(labels)], dim=1)
        return self.classifier(self.features(x)).view(-1)


class ConditionalImageDiscriminator(nn.Module):
    def __init__(self, num_classes=200, image_hw=64):
        super().__init__()
        self.label_map = SpatialLabelMap(num_classes, image_hw, image_hw)
        self.features = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256 * (int(image_hw) // 8) ** 2, 1),
        )

    def forward(self, images, labels):
        x = torch.cat([images, self.label_map(labels)], dim=1)
        return self.classifier(self.features(x)).view(-1)


def set_trainable(module, trainable):
    for parameter in module.parameters():
        parameter.requires_grad_(trainable)
