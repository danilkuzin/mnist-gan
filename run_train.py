import os
from pathlib import Path
import time
from argparse import ArgumentParser

import torch
import torchvision.utils
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from ganmnist.config import GlobalConfig, load_config
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


def load_model(config: GlobalConfig, device):
    if config.model == "vanilla_gan":
        im_dim = (config.dataset.image_size**2) * config.dataset.channels
        gan = GAN(
            config.generator.z_dim,
            im_dim,
            config.generator.num_features,
            config.discriminator.num_features,
        ).to(device)

        gan.gen.apply(init_generator)
        gan.dis.apply(init_discriminator)

    elif config.model == "dcgan":
        gan = DCGAN(
            z_dim=config.generator.z_dim,
            num_channels=config.dataset.channels,
            num_gen_features=config.generator.num_features,
            num_disc_features=config.discriminator.num_features,
            disc_normalization=config.discriminator.normalization,
        ).to(device)

        initialize_weights(gan.dis)
        initialize_weights(gan.gen)
    else:
        raise Exception()

    return gan


def get_optimizers(gan, config):
    OptimClassD = getattr(torch.optim, config.optimizers.discriminator.name)
    OptimClassG = getattr(torch.optim, config.optimizers.generator.name)

    optimizer_d = OptimClassD(
        gan.dis.parameters(), **config.optimizers.discriminator.params
    )
    optimizer_g = OptimClassG(
        gan.gen.parameters(), **config.optimizers.generator.params
    )

    scheduler_d, scheduler_g = None, None

    if getattr(config, "schedulers", None):
        if config.schedulers.discriminator:
            SchedClassD = getattr(
                torch.optim.lr_scheduler, config.schedulers.discriminator.name
            )
            scheduler_d = SchedClassD(
                optimizer_d, **config.schedulers.discriminator.params
            )

        if config.schedulers.generator:
            SchedClassG = getattr(
                torch.optim.lr_scheduler, config.schedulers.generator.name
            )
            scheduler_g = SchedClassG(optimizer_g, **config.schedulers.generator.params)

    return optimizer_d, optimizer_g, scheduler_d, scheduler_g


if __name__ == "__main__":
    max_plot_samples = 32
    num_interp = 10

    parser = ArgumentParser()
    parser.add_argument("-c", "--config")
    args = parser.parse_args()
    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(config.training.seed)

    out_folder = f"/data/logs/{Path(args.config).stem}"
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{config.generator.generator_loss_type}"
    os.makedirs(out_folder, exist_ok=True)
    writer = SummaryWriter(log_dir=f"{out_folder}/{run_id}")

    ds_train, ds_test = load_dataset(config.dataset)
    dl_train = DataLoader(ds_train, shuffle=True, batch_size=config.training.batch_size)

    gan = load_model(config, device)
    optimizer_d, optimizer_g, scheduler_d, scheduler_g = get_optimizers(gan, config)

    if config.model == "vanilla_gan":
        sample_fn = (
            lambda: torch.rand(config.training.batch_size, config.generator.z_dim) * 2
            - 1
        )
        value_range = (0, 1)
        if config.dataset.name == "mnist":
            format_ds = "huggingface"
        else:
            format_ds = "torchvision"
    elif config.model == "dcgan":
        sample_fn = lambda: torch.randn(
            (config.training.batch_size, config.generator.z_dim, 1, 1)
        )
        value_range = (-1, 1)
        if config.dataset.name == "lsun":
            format_ds = "huggingface"
        else:
            format_ds = "torchvision"
    else:
        raise Exception()

    with torch.no_grad():
        z = sample_fn().to(device)
        z1 = sample_fn()[0].to(device)
        z2 = sample_fn()[0].to(device)
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

    optimizer = torch.optim.SGD(gan.gen.parameters(), lr=0.05, momentum=0.5)
    criterion = torch.nn.BCELoss()

    if config.generator.pretrain_epochs > 0:
        for epoch in range(config.generator.pretrain_epochs):
            for real_images, _ in dl_train:
                real_images = real_images.to(device)
                batch_size = real_images.size(0)

                z = torch.randn(batch_size, 100, device=device)
                fake_images = gan.gen(z)
                loss = criterion(fake_images, real_images)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(epoch, loss.item())

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
            config.discriminator.discriminator_loss_type,
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
            config.training.n_critic,
            config.training.weight_clip,
            config.training.lambda_gp,
            device,
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
