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
    pretrain_epochs: int


class DiscriminatorConfig(BaseModel):
    num_features: int


class VisualiseConfig(BaseModel):
    plot_epochs: int
    plot_steps: int


class OptimizerConfig(BaseModel):
    name: str
    params: dict


class SchedulerConfig(BaseModel):
    name: str
    params: dict


class OptimizersConfig(BaseModel):
    discriminator: OptimizerConfig
    generator: OptimizerConfig


class SchedulersConfig(BaseModel):
    discriminator: SchedulerConfig
    generator: SchedulerConfig


class GlobalConfig(BaseModel):
    model: str
    dataset: DatasetConfig
    training: TrainingConfig
    generator: GeneratorConfig
    discriminator: DiscriminatorConfig
    visualise: VisualiseConfig
    optimizers: OptimizersConfig
    schedulers: Optional[SchedulersConfig]


def load_config(path: str):
    with open("configs/" + path + ".yaml", "r") as file:
        data = safe_load(file)

    return GlobalConfig(**data)
