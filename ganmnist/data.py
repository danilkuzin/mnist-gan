import datasets
from ganmnist.config import DatasetConfig
from scipy import io as sio
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset


def load_mnist() -> tuple[datasets.Dataset, datasets.Dataset]:

    ds = datasets.load_dataset("ylecun/mnist", cache_dir="/data/huggingface/datasets")

    preprocess = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1, 28 * 28))]
    )

    def transform_fn(batch):
        batch["image"] = [preprocess(img) for img in batch["image"]]
        batch["label"] = [torch.tensor(label) for label in batch["label"]]
        return batch

    ds.set_transform(transform_fn)

    return ds["train"], ds["test"]


def load_lsun(
    dataset_config: DatasetConfig,
) -> tuple[datasets.Dataset, datasets.Dataset]:

    ds = datasets.load_dataset(
        "pcuenq/lsun-bedrooms", cache_dir="/data/huggingface/datasets"
    )

    preprocess = transforms.Compose(
        [
            transforms.Resize(dataset_config.image_size),
            transforms.CenterCrop(dataset_config.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    def transform_fn(batch):
        batch["image"] = [preprocess(img) for img in batch["image"]]
        return batch

    ds.set_transform(transform_fn)

    return ds["train"], ds["test"]


def load_celeba() -> (
    tuple[torchvision.datasets.VisionDataset, torchvision.datasets.VisionDataset]
):
    ds_transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    ds_train = torchvision.datasets.CelebA(
        root="/data/torchvision_datasets",
        split="train",
        transform=ds_transform,
        download=False,
    )
    ds_test = torchvision.datasets.CelebA(
        root="/data/torchvision_datasets",
        split="test",
        transform=ds_transform,
        download=False,
    )
    return ds_train, ds_test


def load_tfd():
    "based on https://github.com/nouiz/lisa_emotiw/blob/master/emotiw/common/datasets/faces/tfd.py"
    data = sio.loadmat("/data/datasets/TFD/TFD_48x48.mat")
    images = torch.from_numpy(data["images"]).float().reshape(-1, 48 * 48) / 255
    labels = torch.from_numpy(data["labs_ex"]).long()
    dataset = TensorDataset(images, labels)

    return dataset, None


def load_cifar10() -> tuple[datasets.Dataset, datasets.Dataset]:
    ds = datasets.load_dataset(
        "uoft-cs/cifar10", cache_dir="/data/huggingface/datasets"
    )

    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    def transform_fn(batch):
        return {
            "image": [preprocess(img) for img in batch["img"]],
            "labels": [torch.tensor(label) for label in batch["label"]],
        }

    ds.set_transform(transform_fn)

    return ds["train"], ds["test"]


def load_dataset(
    dataset_config: DatasetConfig,
) -> tuple[
    datasets.Dataset | torchvision.datasets.VisionDataset,
    datasets.Dataset | torchvision.datasets.VisionDataset,
]:
    if dataset_config.name == "mnist":
        return load_mnist()
    elif dataset_config.name == "celeba":
        return load_celeba()
    elif dataset_config.name == "tfd":
        return load_tfd()
    elif dataset_config.name == "lsun":
        return load_lsun(dataset_config)
    elif dataset_config.name == "cifar10":
        return load_cifar10()
    else:
        raise Exception()
