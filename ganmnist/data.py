import datasets


def load_mnist() -> tuple[datasets.Dataset, datasets.Dataset]:

    ds = datasets.load_dataset("ylecun/mnist", cache_dir="/data/huggingface/datasets")
    ds.set_format(type="torch")

    return ds["train"], ds["test"]
