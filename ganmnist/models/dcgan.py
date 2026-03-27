import math
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t
from torch.nn.utils import spectral_norm

from ganmnist.config import DatasetConfig, DiscriminatorConfig, GeneratorConfig


class Generator(nn.Module):
    def __init__(
        self,
        data_conf: DatasetConfig,
        gen_conf: GeneratorConfig,
    ) -> None:
        super().__init__()

        self.z_dim = gen_conf.z_dim
        self.num_channels = data_conf.channels
        self.num_features = gen_conf.num_features
        self.conditional_embed_size = gen_conf.conditional_embed_size

        in_ch = self.z_dim
        if self.conditional_embed_size is not None:
            in_ch += self.conditional_embed_size
            self.classes = data_conf.classes
            self.embed = nn.Embedding(self.classes, self.conditional_embed_size)

        num_blocks = int(math.log2(data_conf.image_size)) - 2
        cur_features = self.num_features * (2 ** (num_blocks - 1))

        layers = []
        layers.append(self._block(in_ch, cur_features, 4, 1, 0))
        for _ in range(num_blocks - 1):
            next_features = cur_features // 2
            layers.append(
                self._block(cur_features, next_features, 4, 2, 1),
            )
            cur_features = next_features
        layers.append(nn.ConvTranspose2d(cur_features, self.num_channels, 4, 2, 1))
        layers.append(nn.Tanh())

        self.gen = nn.Sequential(*layers)

    def _block(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: _size_2_t,
        stride: _size_2_t,
        padding: _size_2_t,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, z: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
        if hasattr(self, "embed") and labels is not None:
            # print(f"{labels.shape=} {z.shape=} {self.embed=}")
            # y_embed = self.embed(labels).view(labels.shape[0], -1, 1, 1)
            # z = torch.cat([z, y_embed], dim=1)
            # z = z.unsqueeze(2).unsqueeze(3)
            z = torch.cat([z, self.embed(labels)], dim=1)
        z = z.unsqueeze(2).unsqueeze(3)
        return self.gen(z)


class Discriminator(nn.Module):
    def __init__(
        self,
        data_conf: DatasetConfig,
        dis_conf: DiscriminatorConfig,
    ) -> None:
        super().__init__()

        self.use_spectral_norm = dis_conf.use_spectral_norm
        self.num_channels = data_conf.channels
        self.num_features = dis_conf.num_features
        self.disc_normalization = dis_conf.normalization
        self.conditional = dis_conf.conditional

        num_blocks = int(math.log2(data_conf.image_size)) - 2
        cur_features = self.num_channels

        layers = []
        layers.append(
            self._block(
                cur_features,
                self.num_features,
                kernel_size=4,
                stride=2,
                padding=1,
                disc_normalization=None,
                use_spectral_norm=self.use_spectral_norm,
            )
        )
        cur_features = self.num_features
        for _ in range(num_blocks - 1):
            next_features = cur_features * 2
            layers.append(
                self._block(
                    cur_features,
                    next_features,
                    4,
                    2,
                    1,
                    self.disc_normalization,
                    self.use_spectral_norm,
                )
            )
            cur_features = next_features

        self.blocks = nn.Sequential(*layers)

        self.final_conv = nn.Conv2d(cur_features, 1, 4, 1, 0)
        if self.use_spectral_norm:
            self.final_conv = spectral_norm(self.final_conv)

        if self.conditional and data_conf.classes is not None:
            self.embed = nn.Embedding(self.num_classes, cur_features)
            if self.use_spectral_norm:
                self.embed = spectral_norm(self.embed)

    def _block(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: _size_2_t,
        stride: _size_2_t,
        padding: _size_2_t,
        disc_normalization: Optional[str],
        use_spectral_norm: bool,
    ) -> nn.Sequential:
        if use_spectral_norm and (disc_normalization is not None):
            raise Exception

        layers = []
        conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        if spectral_norm:
            conv = spectral_norm(conv)
        layers.append(conv)

        if disc_normalization is not None:
            NormClass = getattr(nn, disc_normalization)
            layers.append(NormClass(out_ch))

        layers.append(nn.LeakyReLU(0.2))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.blocks(x)
        out = self.final_conv(h).view(-1, 1)
        if self.conditional and labels is not None:
            h_pool = torch.sum(h, dim=[2, 3])
            y_embed = self.embed(labels)
            proj = torch.sum(h_pool * y_embed, dim=1, keepdim=True)

            out += proj
        return out


class DCGAN(nn.Module):
    def __init__(
        self,
        data_conf: DatasetConfig,
        gen_conf: GeneratorConfig,
        dis_conf: DiscriminatorConfig,
    ) -> None:
        super().__init__()

        self.gen = Generator(data_conf, gen_conf)
        self.dis = Discriminator(data_conf, dis_conf)


def initialize_weights(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
