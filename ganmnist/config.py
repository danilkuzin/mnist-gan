from typing import Optional

from pydantic import BaseModel
from yaml import safe_load


class DatasetConfig(BaseModel):
    name: str
    image_size: int
    channels: int


class TrainingConfig(BaseModel):
    epochs: int
    batch_size: int
    seed: int


class GeneratorConfig(BaseModel):
    z_dim: int
    num_features: int
    generator_loss_type: Optional[str] = None


class DiscriminatorConfig(BaseModel):
    num_features: int


class VisualiseConfig(BaseModel):
    plot_epochs: int
    plot_steps: int


class GlobalConfig(BaseModel):
    model: str
    dataset: DatasetConfig
    training: TrainingConfig
    generator: GeneratorConfig
    discriminator: Optional[DiscriminatorConfig] = None
    visualise: VisualiseConfig


def load_config(path: str):
    with open("configs/" + path + ".yaml", "r") as file:
        data = safe_load(file)

    return GlobalConfig(**data)
