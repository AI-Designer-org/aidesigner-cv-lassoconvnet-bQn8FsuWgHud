"""
LassoConvNet — Backbone Architecture.

4-stage ResNet-style pyramid where each block is a LassoProxConvBlock
with integrated soft-thresholding (lasso proximal operator).
"""

import torch.nn as nn
from config import LassoConvConfig
from blocks import LassoProxConvBlock, ListaEncoder


class LassoConvStage(nn.Module):
    """One spatial stage of the pyramid, containing N LassoProxConvBlocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        downsample_first: bool,
        config: LassoConvConfig,
    ):
        super().__init__()
        blocks = []

        for i in range(depth):
            in_ch = in_channels if i == 0 else out_channels
            stride = 2 if (i == 0 and downsample_first) else 1
            block = LassoProxConvBlock(in_ch, out_channels, config)
            blocks.append(block)

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class LassoConvNet(nn.Module):
    """
    LassoConvNet — CNN with integrated Lasso proximal operators.

    Architecture summary:
    - Stem: 3×3 conv, downsample to 1/2
    - Stage 1: C1 channels,  H/2 × W/2,  stride-1 convs
    - Stage 2: C2 channels,  H/4 × W/4,  stride-2 convs
    - Stage 3: C3 channels,  H/8 × W/8,  stride-2 convs
    - Stage 4: C4 channels,  H/16 × W/16, stride-2 convs
    - Pool → FC classifier

    Every convolution is followed by soft-thresholding (lasso proximal op).
    Thresholds θ are per-channel learnable parameters.
    """

    def __init__(self, config: LassoConvConfig):
        super().__init__()
        self.config = config

        # ── Stem ──
        self.stem = nn.Sequential(
            nn.Conv2d(
                config.in_channels,
                config.stage_channels[0],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=config.use_bias,
            ),
            nn.BatchNorm2d(config.stage_channels[0]),
            nn.ReLU(inplace=True),  # standard ReLU in stem; lasso blocks start after
        )

        # ── Stages ──
        self.stages = nn.ModuleList()
        in_ch = config.stage_channels[0]

        for i, (out_ch, depth) in enumerate(
            zip(config.stage_channels, config.stage_depths)
        ):
            downsample_first = (i > 0)  # downsample at start of each stage except first
            stage = LassoConvStage(in_ch, out_ch, depth, downsample_first, config)
            self.stages.append(stage)
            in_ch = out_ch

        # ── LISTA encoder (optional, replaces pyramid) ──
        self.lista_encoder = (
            ListaEncoder(config)
            if config.lasso_mode == "lista_unrolled"
            else None
        )

        # ── Classifier head ──
        last_dim = config.stage_channels[-1]

        if config.lasso_mode == "lista_unrolled":
            last_dim = config.lista_dictionary_size

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(last_dim, config.head_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity(),
            nn.Linear(config.head_hidden_dim, config.n_classes),
        )

    def forward(self, x):
        h = self.stem(x)

        if self.lista_encoder is not None:
            h = self.lista_encoder(h)
        else:
            for stage in self.stages:
                h = stage(h)

        return self.head(h)
