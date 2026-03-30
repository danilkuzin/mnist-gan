import os
from pathlib import Path
import time
from argparse import ArgumentParser

from ganmnist.models import dcgan
from ganmnist.models import vanilla_gan
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
import torchmetrics.image.fid, torchmetrics.image.inception


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


def evaluate(
    gen: dcgan.Generator | vanilla_gan.Generator,
    config: GlobalConfig,
    fid: torchmetrics.image.fid.FrechetInceptionDistance,
    device,
    num_samples: int,
) -> dict[str, float]:
    inception = torchmetrics.image.inception.InceptionScore(
        normalize=True,
    ).to(device)

    batch_size = config.training.batch_size
    num_batches = num_samples // batch_size

    print(f"Generating {num_samples} images for evaluation...")
    for _ in tqdm(range(num_batches), desc="Evaluating", leave=False):
        z = sample_fn().to(device)
        if config.training.conditional:
            y = torch.randint(
                config.dataset.classes, (config.training.batch_size,), device=device
            )
            fake_imgs = gen(z, y)
        else:
            fake_imgs = gen(z)

        if hasattr(gen, "gen") and isinstance(gen.gen[-1], torch.nn.Tanh):
            fake_imgs = (fake_imgs + 1) / 2.0  # [-1, 1] -> [1, 0]

        fid.update(fake_imgs, real=False)
        inception.update(fake_imgs)

    fid_score = fid.compute().item()
    is_mean, is_std = inception.compute()

    fid.reset()
    inception.reset()

    return {"FID": fid_score, "IS_mean": is_mean.item(), "IS_std": is_std.item()}


@torch.no_grad()
def precompute_real_fid(dataloader, device, num_samples=10000):
    fid = torchmetrics.image.fid.FrechetInceptionDistance(
        feature=2048, normalize=True, reset_real_features=False
    ).to(device)

    count = 0
    print("Pre-computing real image statistics for FID...")
    for batch in dataloader:
        real_imgs = batch["image"].to(device)
        real_imgs = (real_imgs + 1) / 2.0

        fid.update(real_imgs, real=True)
        count += real_imgs.shape[0]
        if count >= num_samples:
            break

    return fid


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

    base_fid_metric = precompute_real_fid(dl_train, device, num_samples=10000)

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
                metrics = evaluate(
                    gan.gen, config, base_fid_metric, device, num_samples=10000
                )

                print(f"Epoch {epoch} Evaluation:")
                print(f"FID: {metrics['FID']:.2f} | IS: {metrics['IS_mean']:.2f}")

                writer.add_scalar("Eval/FID", metrics["FID"], epoch)
                writer.add_scalar("Eval/IS", metrics["IS_mean"], epoch)
