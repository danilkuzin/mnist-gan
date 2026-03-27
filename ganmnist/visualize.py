import torch
import torchvision.utils
from torch.utils.tensorboard.writer import SummaryWriter


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
