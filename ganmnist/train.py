import time
from typing import Callable, Optional

from ganmnist.config import GlobalConfig
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter

from ganmnist.models.dcgan import DCGAN
from ganmnist.models.vanilla_gan import GAN
from ganmnist.losses import discriminator_losses, generator_losses


def gradient_penalty(
    critic, real, fake, device, labels: Optional[torch.Tensor]
) -> tuple[torch.Tensor, dict[str, float]]:
    B, C, H, W = real.shape
    epsilon = torch.rand((B, 1, 1, 1), device=device)
    interpolated = real * epsilon + fake * (1 - epsilon)
    interpolated.requires_grad_(True)

    if labels is not None:
        mixed_scores = critic(interpolated, labels)
    else:
        mixed_scores = critic(interpolated)

    gradient = torch.autograd.grad(
        inputs=interpolated,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty_res = torch.mean((gradient_norm - 1) ** 2)
    metrics = {
        "grad penalty": gradient_penalty_res.item(),
        "grad norm": gradient_norm.mean().item(),
    }

    return gradient_penalty_res, metrics


def train_epoch(
    dl: torch.utils.data.DataLoader,
    optimizer_d: torch.optim.Optimizer,
    optimizer_g: torch.optim.Optimizer,
    gan: GAN | DCGAN,
    sample_fn: Callable[[], torch.Tensor],
    format: str,
    z_plotting: torch.Tensor,
    y_plotting: Optional[torch.Tensor],
    plot_fn: Callable[[torch.Tensor, int], None],
    epoch: int,
    device: str,
    config: GlobalConfig,
    writer: SummaryWriter,
):
    batches_per_epoch = len(dl)
    losses_d, losses_g = [], []

    pbar = tqdm(dl, desc="Training", leave=False)
    postfix = {}
    for step, batch in enumerate(pbar):
        global_step = (epoch * batches_per_epoch) + step
        step_start = time.time()

        # train discriminator
        if format == "huggingface":
            x = batch["image"].to(device)
            if config.training.conditional:
                y = batch["labels"].to(device)
        elif format == "torchvision":
            x = batch[0].to(device)
            if config.training.conditional:
                y = batch[1].to(device)
        else:
            raise Exception()

        optimizer_d.zero_grad()

        z_d = sample_fn()[: x.shape[0]].to(device)

        if config.training.conditional:
            g_z = gan.gen(z_d, y)
            disc_real = gan.dis(x, y).view(-1)
            disc_fake = gan.dis(g_z.detach(), y).view(-1)
        else:
            g_z = gan.gen(z_d)
            disc_real = gan.dis(x).view(-1)
            disc_fake = gan.dis(g_z.detach()).view(-1)

        loss_d, metrics_d = discriminator_losses[
            config.discriminator.discriminator_loss_type
        ](disc_real, disc_fake)
        for k, v in metrics_d.items():
            writer.add_scalar(f"D/{k}", v.item(), global_step)
            postfix[f"D/{k}"] = v.item()

        if config.training.lambda_gp is not None:
            if config.training.conditional:
                gp, metrics_gp = gradient_penalty(gan.dis, x, g_z.detach(), device, y)
            else:
                gp, metrics_gp = gradient_penalty(
                    gan.dis, x, g_z.detach(), device, None
                )
            loss_d += config.training.lambda_gp * gp
            for k, v in metrics_gp.items():
                writer.add_scalar(f"D/gp/{k}", v, global_step)
                postfix[f"D/gp/{k}"] = v
            postfix["gp"] = gp.item()  # want to have 0.5-5

        loss_d.backward()
        optimizer_d.step()
        losses_d.append(loss_d.item())
        postfix["D"] = loss_d.item()

        if config.training.weight_clip is not None:
            for p in gan.dis.parameters():
                p.data.clamp_(-config.training.weight_clip, config.training.weight_clip)

        loss_g = None
        if (
            config.training.n_critic is not None
            and step % config.training.n_critic == 0
        ):
            # train generator
            for p in gan.dis.parameters():
                p.requires_grad = False

            optimizer_g.zero_grad()

            z_g = sample_fn().to(device)

            if config.training.conditional:
                y_g = torch.randint(
                    config.dataset.classes, (z_g.shape[0],), device=device
                )
                g_z = gan.gen(z_g, y_g)
                output = gan.dis(g_z, y_g).view(-1)
            else:
                g_z = gan.gen(z_g)
                output = gan.dis(g_z).view(-1)

            loss_g = generator_losses[config.generator.generator_loss_type](output)

            loss_g.backward()
            optimizer_g.step()
            losses_g.append(loss_g.item())
            postfix["G"] = loss_g.item()

            for p in gan.dis.parameters():
                p.requires_grad = True

        step_time = time.time() - step_start
        postfix["t/b"] = f"{step_time*1000:.1f}ms"
        pbar.set_postfix(postfix)

        if config.visualise.plot_steps > 0:
            if global_step % config.visualise.plot_steps == 0:
                if config.training.conditional:
                    generated = gan.gen(z_plotting, y_plotting)
                else:
                    generated = gan.gen(z_plotting)
                plot_fn(generated, global_step)

    print(
        f"{epoch=}. {np.array(losses_d).mean()=:.4f} {np.array(losses_g).mean()=:.4f}"
    )
