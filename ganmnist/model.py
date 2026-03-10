import torch
import torch.nn as nn
import torch.nn.functional as F


class Maxout(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_pieces: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_pieces = num_pieces

        self.linear = nn.Linear(in_features, out_features * num_pieces)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = (x.size(0), self.out_features, self.num_pieces)

        out = self.linear(x)
        out = out.view(*shape)
        out, _ = torch.max(out, dim=2)

        return out


class Generator(nn.Module):
    def __init__(self, z_dim: int, im_dim: int) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.im_dim = im_dim
        self.linear1 = nn.Linear(z_dim, 1200)
        self.linear2 = nn.Linear(1200, 1200)
        self.linear3 = nn.Linear(1200, im_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.linear1(z)
        z = F.relu(z)
        z = self.linear2(z)
        z = F.relu(z)
        z = self.linear3(z)
        z = torch.sigmoid(z)

        return z


class Discriminator(nn.Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.mo1 = Maxout(in_features, 240, 5)
        self.mo2 = Maxout(240, 240, 5)
        self.linear = nn.Linear(240, 1)
        self.d1 = nn.Dropout(0.5)
        self.d2 = nn.Dropout(0.5)
        self.d3 = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.d1(x)
        x = self.mo1(x)
        x = self.d2(x)
        x = self.mo2(x)
        x = self.d3(x)
        x = self.linear(x)

        return x


class GAN(nn.Module):
    def __init__(self, z_dim: int, im_dim: int) -> None:
        super().__init__()
        self.gen = Generator(z_dim, im_dim)
        self.dis = Discriminator(im_dim)


def init_generator(m):
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, -0.05, 0.05)
        nn.init.zeros_(m.bias)


def init_discriminator(m):
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, -0.005, 0.005)
        nn.init.constant_(m.bias, 0.1)
    if isinstance(m, Maxout):
        nn.init.uniform_(m.linear.weight, -0.005, 0.005)
        nn.init.constant_(m.linear.bias, 0.1)
