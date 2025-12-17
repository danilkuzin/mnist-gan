import torch
import torch.nn.functional as F
import numpy as np

from ganmnist.model import GAN


def sample(batch_size, z_dim):
    return torch.randn((batch_size, z_dim))


def train_epoch(
    dl: torch.utils.data.DataLoader,
    optimizer_d: torch.optim.Optimizer,
    optimizer_g: torch.optim.Optimizer,
    gan: GAN,
) -> tuple[float, float, float]:
    gan.train()

    losses_d_real, losses_d_fake, losses_g = [], [], []
    for batch in dl:
        # train discriminator
        x = batch["image"].to("cuda") / 127.5 - 1.0
        x = x.view(-1, 28 * 28 * 1)

        optimizer_d.zero_grad()

        z_d = sample(x.shape[0], gan.gen.z_dim).to("cuda")

        g_z = gan.gen(z_d)
        disc_real = gan.dis(x).view(-1)

        loss_d_real = F.binary_cross_entropy_with_logits(
            disc_real, torch.ones_like(disc_real) * 0.9
        )

        disc_fake = gan.dis(g_z.detach()).view(-1)

        loss_d_fake = F.binary_cross_entropy_with_logits(
            disc_fake, torch.zeros_like(disc_fake)
        )
        loss_d = (loss_d_real + loss_d_fake) / 2

        loss_d.backward()
        optimizer_d.step()
        losses_d_real.append(loss_d_real.item())
        losses_d_fake.append(loss_d_fake.item())

        # train generator
        optimizer_g.zero_grad()

        z_g = sample(x.shape[0], gan.gen.z_dim).to("cuda")

        g_z = gan.gen(z_g)
        output = gan.dis(g_z)

        loss_g = F.binary_cross_entropy_with_logits(output, torch.ones_like(output))

        loss_g.backward()
        optimizer_g.step()
        losses_g.append(loss_g.item())

    return (
        np.array(losses_d_real).mean(),
        np.array(losses_d_fake).mean(),
        np.array(losses_g).mean(),
    )
