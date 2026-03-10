import os
import time

import matplotlib.pyplot as plt
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from ganmnist.data import load_mnist
from ganmnist.model import GAN, init_discriminator, init_generator
from ganmnist.train import train_epoch, sample


def save_grid(
    generated: torch.Tensor, path: str, max_plot_samples: int, nrows: int, ncols: int
):
    n_samples = generated.shape[0]
    grid = generated.detach().cpu().numpy()
    for i in range(min(n_samples, max_plot_samples)):
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(grid[i, 0], cmap="gray")
        plt.axis("off")
    plt.savefig(path)
    plt.close("all")


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


if __name__ == "__main__":
    z_dim = 100
    epochs = 200
    batch_size = 100
    generator_loss_type = "non_saturating"
    max_plot_samples = 32
    num_interp = 10
    seed = 100
    out_folder = "/data/logs/gan/"

    torch.manual_seed(seed)
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{generator_loss_type}"
    os.makedirs(out_folder, exist_ok=True)
    writer = SummaryWriter(log_dir=f"{out_folder}/{run_id}")

    ds_train, ds_test = load_mnist()
    dl_train = DataLoader(ds_train, shuffle=True, batch_size=batch_size)

    im_dim = 28 * 28 * 1
    gan = GAN(z_dim, im_dim).to("cuda")

    gan.gen.apply(init_generator)
    gan.dis.apply(init_discriminator)

    optimizer_d = SGD(list(gan.dis.parameters()), lr=1e-1, momentum=0.5)
    optimizer_g = SGD(list(gan.gen.parameters()), lr=1e-1, momentum=0.5)
    scheduler_d = ExponentialLR(optimizer_d, gamma=1 / 1.000004)
    scheduler_g = ExponentialLR(optimizer_g, gamma=1 / 1.000004)

    pixel_mean = compute_pixel_mean(ds_train)
    bias = torch.log(pixel_mean / (1 - pixel_mean))
    gan.gen.linear3.bias.data.fill_(bias)

    with torch.no_grad():
        z = sample(batch_size, z_dim).to("cuda")
        z1 = sample(1, z_dim).to("cuda")
        z2 = sample(1, z_dim).to("cuda")
        z_interp = interpolate(z1, z2, num_interp).squeeze(1)

        generated = gan.gen(z)
        generated = generated.view(-1, 1, 28, 28)
        save_grid(
            generated,
            f"{out_folder}/{run_id}/samples_iter_-1.png",
            max_plot_samples,
            4,
            8,
        )

        interpolated = gan.gen(z_interp)
        interpolated = interpolated.view(-1, 1, 28, 28)
        save_grid(
            interpolated,
            f"{out_folder}/{run_id}/interpolated_iter_-1.png",
            num_interp,
            1,
            num_interp,
        )

    for epoch in range(epochs):
        gan.gen.train()
        gan.dis.train()
        train_loss_d_real, train_loss_d_fake, train_loss_g = train_epoch(
            dl_train, optimizer_d, optimizer_g, gan, generator_loss_type
        )
        scheduler_d.step()
        scheduler_g.step()
        writer.add_scalar("D/real", train_loss_d_real, epoch)
        writer.add_scalar("D/fake", train_loss_d_fake, epoch)
        writer.add_scalar("G", train_loss_g, epoch)

        if epoch % 10 == 0:
            gan.gen.eval()
            gan.dis.eval()
            with torch.no_grad():
                generated = gan.gen(z)
                generated = generated.view(-1, 1, 28, 28)
                save_grid(
                    generated,
                    f"{out_folder}/{run_id}/samples_iter_{epoch}.png",
                    max_plot_samples,
                    4,
                    8,
                )

                interpolated = gan.gen(z_interp)
                interpolated = interpolated.view(-1, 1, 28, 28)
                save_grid(
                    interpolated,
                    f"{out_folder}/{run_id}/interpolated_iter_{epoch}.png",
                    num_interp,
                    1,
                    num_interp,
                )

        print(
            f"{epoch=}. {train_loss_d_real=:.4f} {train_loss_d_fake=:.4f} {train_loss_g=:.4f}"
        )
