import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, z_dim: int, im_dim: int) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.im_dim = im_dim
        self.linear1 = nn.Linear(z_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.linear2 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.linear3 = nn.Linear(512, im_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.linear1(z)
        z = self.bn1(z)
        z = F.leaky_relu(z, 0.1)
        z = self.linear2(z)
        z = self.bn2(z)
        z = F.leaky_relu(z, 0.1)
        z = self.linear3(z)
        z = F.tanh(z)

        return z


class Discriminator(nn.Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.linear1 = nn.Linear(self.in_features, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.leaky_relu(x, 0.1)
        x = self.linear2(x)
        x = F.leaky_relu(x, 0.1)
        x = self.linear3(x)

        return x


class GAN(nn.Module):
    def __init__(self, z_dim: int, im_dim: int) -> None:
        super().__init__()
        self.gen = Generator(z_dim, im_dim)
        self.dis = Discriminator(im_dim)
