import time
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ganmnist.models.dcgan import DCGAN
from ganmnist.models.vanilla_gan import GAN


def train_epoch(
    dl: torch.utils.data.DataLoader,
    optimizer_d: torch.optim.Optimizer,
    optimizer_g: torch.optim.Optimizer,
    gan: GAN | DCGAN,
    generator_loss_type: str,
    sample_fn: Callable[[], torch.Tensor],
    format: str,
    plot_steps: int,
    z_plotting: torch.Tensor,
    plot_fn: Callable[[torch.Tensor, int], None],
    last_iter: int,
) -> tuple[float, float, float, int]:

    losses_d_real, losses_d_fake, losses_g = [], [], []
    pbar = tqdm(dl, desc="Training", leave=False)
    for step, batch in enumerate(pbar):
        step_start = time.time()
        # train discriminator
        if format == "huggingface":
            x = batch["image"].to("cuda")
        elif format == "torchvision":
            x = batch[0].to("cuda")
        else:
            raise Exception()

        optimizer_d.zero_grad()

        z_d = sample_fn().to("cuda")

        g_z = gan.gen(z_d)
        disc_real = gan.dis(x).view(-1)

        loss_d_real = F.binary_cross_entropy_with_logits(
            disc_real, torch.ones_like(disc_real)
        )

        disc_fake = gan.dis(g_z.detach()).view(-1)

        loss_d_fake = F.binary_cross_entropy_with_logits(
            disc_fake, torch.zeros_like(disc_fake)
        )
        loss_d = loss_d_real + loss_d_fake  # /2 potentially for dcgan?

        loss_d.backward()
        optimizer_d.step()
        losses_d_real.append(loss_d_real.item())
        losses_d_fake.append(loss_d_fake.item())

        # train generator
        for p in gan.dis.parameters():
            p.requires_grad = False

        optimizer_g.zero_grad()

        z_g = sample_fn().to("cuda")

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

        step_time = time.time() - step_start
        pbar.set_postfix(
            {
                "D_real": f"{loss_d_real.item():.3f}",
                "D_fake": f"{loss_d_fake.item():.3f}",
                "G": f"{loss_g.item():.3f}",
                "t/b": f"{step_time*1000:.1f}ms",
            }
        )

        if plot_steps > 0:
            if step % plot_steps == 0:
                generated = gan.gen(z_plotting)
                plot_fn(generated, step)
        last_iter += 1

    return (
        np.array(losses_d_real).mean(),
        np.array(losses_d_fake).mean(),
        np.array(losses_g).mean(),
        last_iter,
    )
