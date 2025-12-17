import os
import time

import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from ganmnist.data import load_mnist
from ganmnist.model import GAN
from ganmnist.train import train_epoch, sample


def save_grid(generated, iter):
    n_samples = generated.shape[0]
    grid = generated.detach().cpu().numpy()
    for i in range(n_samples):
        plt.subplot(4, 8, i + 1)
        plt.imshow((grid[i, 0] + 1) * 127.5, cmap="gray")
        plt.axis("off")
    plt.savefig(f"/data/logs/gan/samples_iter_{iter}.png")
    plt.close("all")


if __name__ == "__main__":
    z_dim = 64
    epochs = 50  # 50
    batch_size = 32

    out_folder = "/data/logs/gan/"
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(out_folder, exist_ok=True)
    seed = 100
    torch.manual_seed(seed)

    writer = SummaryWriter(log_dir=f"{out_folder}/{run_id}")

    ds_train, ds_test = load_mnist()
    dl_train = DataLoader(ds_train, shuffle=True, batch_size=batch_size)

    im_dim = 28 * 28 * 1
    gan = GAN(z_dim, im_dim).to("cuda")

    optimizer_d = Adam(list(gan.dis.parameters()), lr=1e-4, betas=(0.5, 0.999))
    optimizer_g = Adam(list(gan.gen.parameters()), lr=3e-4, betas=(0.5, 0.999))

    with torch.no_grad():
        z = sample(batch_size, z_dim).to("cuda")
        generated = gan.gen(z)
        generated = generated.view(-1, 1, 28, 28)
        save_grid(generated, -1)

    for epoch in range(epochs):
        train_loss_d_real, train_loss_d_fake, train_loss_g = train_epoch(
            dl_train, optimizer_d, optimizer_g, gan
        )
        writer.add_scalar("D/real", train_loss_d_real, epoch)
        writer.add_scalar("D/fake", train_loss_d_fake, epoch)
        writer.add_scalar("G", train_loss_g, epoch)

        with torch.no_grad():
            generated = gan.gen(z)
            generated = generated.view(-1, 1, 28, 28)
            save_grid(generated, epoch)

        print(
            f"{epoch=}. {train_loss_d_real=:.4f} {train_loss_d_fake=:.4f} {train_loss_g=:.4f}"
        )
