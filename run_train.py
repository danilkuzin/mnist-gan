import os
from pathlib import Path
import time
from argparse import ArgumentParser

from ganmnist.visualize import interpolate, save_grid
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
        gan = DCGAN(config.dataset, config.generator, config.discriminator).to(device)

        initialize_weights(gan.dis)
        initialize_weights(gan.gen)
    else:
        raise Exception()

    return gan


def get_optimizers(gan: GAN | DCGAN, config: GlobalConfig):
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
            (config.training.batch_size, config.generator.z_dim)
        )
        value_range = (-1, 1)
        if config.dataset.name == "lsun" or config.dataset.name == "cifar10":
            format_ds = "huggingface"
        else:
            format_ds = "torchvision"
    else:
        raise Exception()

    with torch.no_grad():
        z_plotting = sample_fn().to(device)
        z1_plotting = sample_fn()[0].to(device)
        z2_plotting = sample_fn()[0].to(device)
        y_plotting = None

        if config.training.conditional:
            y_plotting = torch.randint(
                config.dataset.classes, (config.training.batch_size,), device=device
            )
            y_interp = (
                torch.randint(config.dataset.classes, (1,), device=device)
                .repeat(num_interp)
                .squeeze()
            )

        z_interp = interpolate(z1_plotting, z2_plotting, num_interp).squeeze(1)

        if config.training.conditional:
            generated = gan.gen(z_plotting, y_plotting)
        else:
            generated = gan.gen(z_plotting)
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

        if config.training.conditional:
            interpolated = gan.gen(z_interp, y_interp)
        else:
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
            num_interp,
            value_range,
        )

    optimizer = torch.optim.SGD(gan.gen.parameters(), lr=0.05, momentum=0.5)
    criterion = torch.nn.BCELoss()

    if config.model == "vanilla_gan" and config.generator.pretrain_epochs > 0:
        for epoch in range(config.generator.pretrain_epochs):
            for real_images, _ in dl_train:
                real_images = real_images.to(device)
                batch_size = real_images.size(0)

                z = torch.randn(batch_size, config.generator.z_dim, device=device)
                y = torch.randint(
                    config.dataset.classes, (config.training.batch_size,), device=device
                )
                if config.training.conditional:
                    fake_images = gan.gen(z, y)
                else:
                    fake_images = gan.gen(z)
                loss = criterion(fake_images, real_images)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(epoch, loss.item())

    # last_iter = 0
    for epoch in range(config.training.epochs):
        gan.gen.train()
        gan.dis.train()
        train_epoch(
            dl_train,
            optimizer_d,
            optimizer_g,
            gan,
            sample_fn,
            format_ds,
            z_plotting,
            y_plotting,
            lambda g, b: save_grid(
                g.view(
                    -1,
                    config.dataset.channels,
                    config.dataset.image_size,
                    config.dataset.image_size,
                ),
                writer,
                "Generated",
                b,
                16,
                value_range,
            ),
            epoch,
            device,
            config,
            writer,
        )
        if not scheduler_d is None:
            scheduler_d.step()
        if not scheduler_g is None:
            scheduler_g.step()

        if epoch % config.visualise.plot_epochs == 0:
            gan.gen.eval()
            gan.dis.eval()
            with torch.no_grad():
                if config.training.conditional:
                    generated = gan.gen(z_plotting, y_plotting)
                else:
                    generated = gan.gen(z_plotting)
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
                    epoch * len(dl_train),
                    16,
                    value_range,
                )

                if config.training.conditional:
                    interpolated = gan.gen(z_interp, y_interp)
                else:
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
                    epoch * len(dl_train),
                    num_interp,
                    value_range,
                )
