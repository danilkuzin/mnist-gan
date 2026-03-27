import torch
import torch.nn.functional as F


def sum_log_loss(
    disc_real: torch.Tensor, disc_fake: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # vanilla gan
    loss_d_real = F.binary_cross_entropy_with_logits(
        disc_real, torch.ones_like(disc_real)
    )
    loss_d_fake = F.binary_cross_entropy_with_logits(
        disc_fake, torch.zeros_like(disc_fake)
    )

    loss_d = loss_d_real + loss_d_fake

    metrics = {
        "real_loss": loss_d_real.detach(),
        "fake_loss": loss_d_fake.detach(),
    }

    return loss_d, metrics


def half_sum_log_loss(
    disc_real: torch.Tensor, disc_fake: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # dcgan
    loss_d_real = F.binary_cross_entropy_with_logits(
        disc_real, torch.ones_like(disc_real)
    )
    loss_d_fake = F.binary_cross_entropy_with_logits(
        disc_fake, torch.zeros_like(disc_fake)
    )

    loss_d = (loss_d_real + loss_d_fake) / 2

    metrics = {
        "real_loss": loss_d_real.detach(),
        "fake_loss": loss_d_fake.detach(),
        "real_score": torch.sigmoid(disc_real).detach(),
        "fake_score": torch.sigmoid(disc_fake).detach(),
    }
    return loss_d, metrics


def mean_loss(
    disc_real: torch.Tensor, disc_fake: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # wgan
    loss_d_real = -torch.mean(disc_real)
    loss_d_fake = torch.mean(disc_fake)

    loss_d = loss_d_real + loss_d_fake

    metrics = {
        "real_score": -loss_d_real.detach(),
        "fake_score": loss_d_fake.detach(),
    }

    return loss_d, metrics


def hinge_loss(
    disc_real: torch.Tensor, disc_fake: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_d = torch.mean(F.relu(1 - disc_real)) + torch.mean(F.relu(1 + disc_fake))

    return loss_d, {}


discriminator_losses = {
    "sum_log": sum_log_loss,
    "half_sum_log": half_sum_log_loss,
    "mean": mean_loss,
    "hinge": hinge_loss,
}


def non_saturating_loss(output: torch.Tensor) -> torch.Tensor:
    # vanilla gan
    return F.binary_cross_entropy_with_logits(output, torch.ones_like(output))


def minimax_loss(output: torch.Tensor) -> torch.Tensor:
    # vanilla gan
    return -F.binary_cross_entropy_with_logits(output, torch.zeros_like(output))


def wgan_loss(output: torch.Tensor) -> torch.Tensor:
    # wgan
    return -torch.mean(output)


generator_losses = {
    "non_saturating": non_saturating_loss,
    "minimax": minimax_loss,
    "wgan": wgan_loss,
}
