import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t
from torch.nn.utils import spectral_norm
from typing import Optional


class Generator(nn.Module):
    def __init__(self, z_dim: int, num_channels: int, num_features: int) -> None:
        super().__init__()

        self.z_dim = z_dim

        self.gen = nn.Sequential(
            self._block(z_dim, num_features * 16, 4, 1, 0),
            self._block(num_features * 16, num_features * 8, 4, 2, 1),
            self._block(num_features * 8, num_features * 4, 4, 2, 1),
            self._block(num_features * 4, num_features * 2, 4, 2, 1),
            nn.ConvTranspose2d(num_features * 2, num_channels, 4, 2, 1),
            nn.Tanh(),
        )

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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.gen(z)


class Discriminator(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_features: int,
        disc_normalization: Optional[str],
        use_spectral_norm: Optional[bool],
    ) -> None:
        super().__init__()

        if use_spectral_norm is None:
            use_spectral_norm = False

        self.disc = nn.Sequential(
            self._block(
                num_channels,
                num_features,
                kernel_size=4,
                stride=2,
                padding=1,
                disc_normalization=None,
                use_spectral_norm=use_spectral_norm,
            ),
            self._block(
                num_features,
                num_features * 2,
                4,
                2,
                1,
                disc_normalization,
                use_spectral_norm,
            ),
            self._block(
                num_features * 2,
                num_features * 4,
                4,
                2,
                1,
                disc_normalization,
                use_spectral_norm,
            ),
            self._block(
                num_features * 4,
                num_features * 8,
                4,
                2,
                1,
                disc_normalization,
                use_spectral_norm,
            ),
        )
        if use_spectral_norm:
            self.disc.append(spectral_norm(nn.Conv2d(num_features * 8, 1, 4, 1, 0)))
        else:
            self.disc.append(nn.Conv2d(num_features * 8, 1, 4, 1, 0))

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

        if use_spectral_norm:
            return nn.Sequential(
                spectral_norm(
                    nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
                ),
                nn.LeakyReLU(0.2),
            )
        else:
            if disc_normalization is not None:
                NormClass = getattr(nn, disc_normalization)

                return nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
                    NormClass(out_ch),
                    nn.LeakyReLU(0.2),
                )
            else:
                return nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
                    nn.LeakyReLU(0.2),
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.disc(x)


class DCGAN(nn.Module):
    def __init__(
        self,
        z_dim: int,
        num_channels: int,
        num_gen_features: int,
        num_disc_features: int,
        disc_normalization: Optional[str],
        disc_use_spectral_norm: Optional[bool],
    ) -> None:
        super().__init__()
        self.gen = Generator(z_dim, num_channels, num_gen_features)
        self.dis = Discriminator(
            num_channels, num_disc_features, disc_normalization, disc_use_spectral_norm
        )


def initialize_weights(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
