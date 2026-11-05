import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return nn.functional.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Block(nn.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim))
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return residual + self.drop_path(x)


class ConvNeXtBackbone(nn.Module):
    def __init__(
        self,
        in_chans=3,
        num_classes=200,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
    ):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )

        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cur = 0
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward_features_to_cut(self, x, cut_idx):
        for i in range(cut_idx):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x

    def forward_from_cut(self, x, cut_idx):
        for i in range(cut_idx, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = x.mean(dim=(-2, -1))
        x = self.norm(x)
        return self.head(x)

    def forward(self, x):
        x = self.forward_features_to_cut(x, 4)
        x = x.mean(dim=(-2, -1))
        x = self.norm(x)
        return self.head(x)


def cut_to_index(cut):
    cut = cut.lower()
    mapping = {"c1": 1, "c2": 2, "c3": 3, "c4": 4}
    if cut not in mapping:
        raise ValueError(f"Invalid cut {cut}; expected one of {sorted(mapping)}")
    return mapping[cut]


class ConvNeXtClient(nn.Module):
    def __init__(self, backbone, cut):
        super().__init__()
        self.cut = cut.lower()
        self.cut_idx = cut_to_index(cut)
        self.downsample_layers = nn.ModuleList(backbone.downsample_layers[: self.cut_idx])
        self.stages = nn.ModuleList(backbone.stages[: self.cut_idx])

    def forward(self, x):
        for i in range(self.cut_idx):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x


class ConvNeXtServer(nn.Module):
    def __init__(self, backbone, cut):
        super().__init__()
        self.cut = cut.lower()
        self.cut_idx = cut_to_index(cut)
        self.downsample_layers = nn.ModuleList(backbone.downsample_layers[self.cut_idx :])
        self.stages = nn.ModuleList(backbone.stages[self.cut_idx :])
        self.norm = backbone.norm
        self.head = backbone.head

    def forward(self, x):
        for downsample, stage in zip(self.downsample_layers, self.stages):
            x = downsample(x)
            x = stage(x)
        x = x.mean(dim=(-2, -1))
        x = self.norm(x)
        return self.head(x)


def build_split_convnext_tiny(cut, num_classes=200, drop_path_rate=0.0):
    backbone = ConvNeXtBackbone(
        num_classes=num_classes,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        drop_path_rate=drop_path_rate,
    )
    return ConvNeXtClient(backbone, cut), ConvNeXtServer(backbone, cut)


def expected_activation_shape(cut):
    shapes = {
        "c1": (96, 16, 16),
        "c2": (192, 8, 8),
        "c3": (384, 4, 4),
        "c4": (768, 2, 2),
    }
    return shapes[cut.lower()]
