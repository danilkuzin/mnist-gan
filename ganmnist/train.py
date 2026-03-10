import torch
import torch.nn.functional as F
import numpy as np

from ganmnist.model import GAN


def sample(batch_size, z_dim):
    return torch.rand(batch_size, z_dim) * 2 - 1


def train_epoch(
    dl: torch.utils.data.DataLoader,
    optimizer_d: torch.optim.Optimizer,
    optimizer_g: torch.optim.Optimizer,
    gan: GAN,
    generator_loss_type: str,
) -> tuple[float, float, float]:

    losses_d_real, losses_d_fake, losses_g = [], [], []
    for batch in dl:
        # train discriminator
        x = batch["image"].to("cuda") / 255
        x = x.view(-1, 28 * 28 * 1)

        optimizer_d.zero_grad()

        z_d = sample(x.shape[0], gan.gen.z_dim).to("cuda")

        g_z = gan.gen(z_d)
        disc_real = gan.dis(x).view(-1)

        loss_d_real = F.binary_cross_entropy_with_logits(
            disc_real, torch.ones_like(disc_real)
        )

        disc_fake = gan.dis(g_z.detach()).view(-1)

        loss_d_fake = F.binary_cross_entropy_with_logits(
            disc_fake, torch.zeros_like(disc_fake)
        )
        loss_d = loss_d_real + loss_d_fake

        loss_d.backward()
        optimizer_d.step()
        losses_d_real.append(loss_d_real.item())
        losses_d_fake.append(loss_d_fake.item())

        # train generator
        for p in gan.dis.parameters():
            p.requires_grad = False

        optimizer_g.zero_grad()

        z_g = sample(x.shape[0], gan.gen.z_dim).to("cuda")

        g_z = gan.gen(z_g)
        output = gan.dis(g_z)

        if generator_loss_type == "non_saturating":
            loss_g = F.binary_cross_entropy_with_logits(output, torch.ones_like(output))
        elif generator_loss_type == "minimax":
            loss_g = -F.binary_cross_entropy_with_logits(
                output, torch.zeros_like(output)
            )
        else:
            raise Exception("Wrong generator loss type")

        loss_g.backward()
        optimizer_g.step()
        losses_g.append(loss_g.item())

        for p in gan.dis.parameters():
            p.requires_grad = True

    return (
        np.array(losses_d_real).mean(),
        np.array(losses_d_fake).mean(),
        np.array(losses_g).mean(),
    )
