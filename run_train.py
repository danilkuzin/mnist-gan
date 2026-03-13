import os
import time
from argparse import ArgumentParser

import torch
import torchvision.utils
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from ganmnist.config import load_config
from ganmnist.data import load_dataset
from ganmnist.models.dcgan import DCGAN, initialize_weights
from ganmnist.models.vanilla_gan import GAN, init_discriminator, init_generator
from ganmnist.train import train_epoch


def save_grid(
    generated: torch.Tensor,
    writer: SummaryWriter,
    tag: str,
    step: int,
    nrows: int,
    value_range: tuple[int, int],
):
    grid = generated.detach().cpu().numpy()
    grid = torchvision.utils.make_grid(
        generated,
        nrow=nrows,
        normalize=True,
        value_range=value_range,
    )
    writer.add_image(tag, grid, global_step=step)


def interpolate(z1, z2, steps):
    alphas = torch.linspace(0, 1, steps, device=z1.device)
    return torch.stack([(1 - a) * z1 + a * z2 for a in alphas])


def compute_pixel_mean(dataset):
    total = 0.0
    count = 0

    for sample in tqdm(dataset):
        img = sample["image"].float()
        total += img.sum()
        count += img.numel()

    return (total / count) / 255.0


def load_model(config):
    if config.model == "vanilla_gan":
        im_dim = (config.dataset.image_size**2) * config.dataset.channels
        gan = GAN(config.generator.z_dim, im_dim, config.generator.num_features).to(
            "cuda"
        )

        gan.gen.apply(init_generator)
        gan.dis.apply(init_discriminator)

    elif config.model == "dcgan":
        gan = DCGAN(
            z_dim=config.generator.z_dim,
            num_channels=config.dataset.channels,
            num_gen_features=config.generator.num_features,
            num_disc_features=config.discriminator.num_features,
        ).to("cuda")

        initialize_weights(gan.dis)
        initialize_weights(gan.gen)
    else:
        raise Exception()

    return gan


def get_optimizers(gan, config):
    if config.model == "vanilla_gan":
        optimizer_d = SGD(list(gan.dis.parameters()), lr=1e-1, momentum=0.5)
        optimizer_g = SGD(list(gan.gen.parameters()), lr=1e-1, momentum=0.5)
        scheduler_d = ExponentialLR(optimizer_d, gamma=1 / 1.000004)
        scheduler_g = ExponentialLR(optimizer_g, gamma=1 / 1.000004)
    elif config.model == "dcgan":
        optimizer_d = Adam(list(gan.dis.parameters()), lr=1e-4, betas=(0.5, 0.999))
        optimizer_g = Adam(list(gan.gen.parameters()), lr=3e-4, betas=(0.5, 0.999))
        scheduler_d = None
        scheduler_g = None
    else:
        raise Exception()

    return optimizer_d, optimizer_g, scheduler_d, scheduler_g


if __name__ == "__main__":
    max_plot_samples = 32
    num_interp = 10

    parser = ArgumentParser()
    parser.add_argument("-c", "--config")
    args = parser.parse_args()
    config = load_config(args.config)

    torch.manual_seed(config.training.seed)

    out_folder = f"/data/logs/{config.model}/{config.dataset.name}"
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{config.generator.generator_loss_type}"
    os.makedirs(out_folder, exist_ok=True)
    writer = SummaryWriter(log_dir=f"{out_folder}/{run_id}")

    ds_train, ds_test = load_dataset(config.dataset.name)
    dl_train = DataLoader(ds_train, shuffle=True, batch_size=config.training.batch_size)

    gan = load_model(config)
    optimizer_d, optimizer_g, scheduler_d, scheduler_g = get_optimizers(gan, config)

    if config.model == "vanilla_gan":
        sample_fn = (
            lambda: torch.rand(config.training.batch_size, config.generator.z_dim) * 2
            - 1
        )
        value_range = (0, 1)
        format_ds = "huggingface"
    elif config.model == "dcgan":
        sample_fn = lambda: torch.randn(
            (config.training.batch_size, config.generator.z_dim, 1, 1)
        )
        value_range = (-1, 1)
        format_ds = "torchvision"
    else:
        raise Exception()

    with torch.no_grad():
        z = sample_fn().to("cuda")
        z1 = sample_fn()[0].to("cuda")
        z2 = sample_fn()[0].to("cuda")
        z_interp = interpolate(z1, z2, num_interp).squeeze(1)

        generated = gan.gen(z)
        generated = generated.view(
            -1,
            config.dataset.channels,
            config.dataset.image_size,
            config.dataset.image_size,
        )
        save_grid(
            generated,
            writer,
            "Generated",
            0,
            16,
            value_range,
        )

        interpolated = gan.gen(z_interp)
        interpolated = interpolated.view(
            -1,
            config.dataset.channels,
            config.dataset.image_size,
            config.dataset.image_size,
        )
        save_grid(
            interpolated,
            writer,
            "Interpolated",
            0,
            1,
            value_range,
        )

    last_iter = 0
    for epoch in range(config.training.epochs):
        gan.gen.train()
        gan.dis.train()
        train_loss_d_real, train_loss_d_fake, train_loss_g, last_iter = train_epoch(
            dl_train,
            optimizer_d,
            optimizer_g,
            gan,
            config.generator.generator_loss_type,
            sample_fn,
            format_ds,
            config.visualise.plot_steps,
            z,
            lambda g, b: save_grid(
                g.view(
                    -1,
                    config.dataset.channels,
                    config.dataset.image_size,
                    config.dataset.image_size,
                ),
                writer,
                "Generated",
                last_iter + b,
                16,
                value_range,
            ),
            last_iter,
        )
        if not scheduler_d is None:
            scheduler_d.step()
        if not scheduler_g is None:
            scheduler_g.step()
        writer.add_scalar("D/real", train_loss_d_real, epoch)
        writer.add_scalar("D/fake", train_loss_d_fake, epoch)
        writer.add_scalar("G", train_loss_g, epoch)

        if epoch % config.visualise.plot_epochs == 0:
            gan.gen.eval()
            gan.dis.eval()
            with torch.no_grad():
                generated = gan.gen(z)
                generated = generated.view(
                    -1,
                    config.dataset.channels,
                    config.dataset.image_size,
                    config.dataset.image_size,
                )
                save_grid(
                    generated,
                    writer,
                    "Generated",
                    last_iter,
                    16,
                    value_range,
                )

                interpolated = gan.gen(z_interp)
                interpolated = interpolated.view(
                    -1,
                    config.dataset.channels,
                    config.dataset.image_size,
                    config.dataset.image_size,
                )
                save_grid(
                    interpolated,
                    writer,
                    "Interpolated",
                    last_iter,
                    1,
                    value_range,
                )

        print(
            f"{epoch=}. {train_loss_d_real=:.4f} {train_loss_d_fake=:.4f} {train_loss_g=:.4f}"
        )
