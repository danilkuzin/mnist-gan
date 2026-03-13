import datasets
import torch
import torchvision
import torchvision.transforms as transforms


def load_mnist() -> tuple[datasets.Dataset, datasets.Dataset]:

    ds = datasets.load_dataset("ylecun/mnist", cache_dir="/data/huggingface/datasets")

    preprocess = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1, 28 * 28))]
    )

    def transform_fn(batch):
        batch["image"] = [preprocess(img) for img in batch["image"]]
        batch["label"] = [torch.tensor(l) for l in batch["label"]]
        return batch

    ds.set_transform(transform_fn)

    return ds["train"], ds["test"]


def load_celeba() -> (
    tuple[torchvision.datasets.VisionDataset, torchvision.datasets.VisionDataset]
):
    ds_transform = torchvision.transforms.Compose(
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


def load_dataset(
    ds_name: str,
) -> tuple[
    datasets.Dataset | torchvision.datasets.VisionDataset,
    datasets.Dataset | torchvision.datasets.VisionDataset,
]:
    if ds_name == "mnist":
        return load_mnist()
    elif ds_name == "celeba":
        return load_celeba()
    else:
        raise Exception()
