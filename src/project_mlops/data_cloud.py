# import typer
import glob
import os

import numpy as np
import torch
from torchvision.io import read_image as img2torch


def fetch_kaggle(forced: bool = False) -> None:
    """Download data from kaggle and save it to raw directory"""
    kaggle_url = "gpiosenka/cards-image-datasetclassification"
    raw_dir = "data/raw/cards-dataset"
    print(
        "In order to download with kaggle.api, it requires kaggle authentication.\n  See this for HowTo: https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md#api-credentials"
    )
    if len(glob.glob(raw_dir + "**/*.*")) > 0 and not forced:
        print("Directory already exist! Aborting download.\n  Enable 'forced' to still proceed with download!")
        return
    # download
    kg.api.authenticate()
    kg.api.dataset_download_files(dataset=kaggle_url, path=f"{raw_dir}", force=forced, unzip=True)
    print("Done downloading")


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images."""
    return (images - images.mean()) / images.std()


def preprocess_data() -> None:
    """Process raw data and save it to processed directory."""
    raw_dir = "data/raw/cards-dataset"
    processed_dir = "data/processed/cards-dataset"
    # If directory does not exist, make one
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    # extract class as dict
    sort_order = {
        v: f"{i:0>2}" for i, v in enumerate("ace two three four five six seven eight nine ten jack queen king".split())
    }
    sort_condition = lambda c: c.split()[-1] + sort_order[c.split()[0]]
    targets_set = {c.replace("\\", "/").split("/")[-1] for c in glob.glob("data/raw/cards-dataset" + "/train/*")}
    targets_sort = sorted(targets_set - {"joker"}, key=sort_condition) + [
        "joker"
    ]  # remove joker, sort list and then add joker back in
    idx2targets = dict(enumerate(targets_sort))
    targets2idx = dict(map(reversed, idx2targets.items()))
    torch.save(idx2targets, f"{processed_dir}/label_converter.pt")
    np.save(f"{processed_dir}/label_converter.npy", idx2targets)

    # preprocess data
    for data in ["train", "valid", "test"]:
        print("Preprocessing", data)
        images_list, target_list = [], []
        for img_path in glob.glob(f"{raw_dir}/{data}/*/*.jpg"):
            img_path = img_path.replace("\\", "/")
            images_list.append(img2torch(img_path))
            target_list.append(targets2idx[img_path.split("/")[-2]])
        # apply normalize
        images_list = normalize(torch.stack(images_list).float())
        target_list = torch.as_tensor(target_list).long()
        torch.save(images_list, f"{processed_dir}/{data}_images.pt")
        torch.save(target_list, f"{processed_dir}/{data}_target.pt")
    print("Done preprocessing")


def playing_cards(project_dir=None) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test datasets for playing-cards dataset."""
    train_images = torch.load(f"{project_dir}/data/processed/cards-dataset/train_images.pt", weights_only=True)
    train_target = torch.load(f"{project_dir}/data/processed/cards-dataset/train_target.pt", weights_only=True)
    valid_images = torch.load(f"{project_dir}/data/processed/cards-dataset/valid_images.pt", weights_only=True)
    valid_target = torch.load(f"{project_dir}/data/processed/cards-dataset/valid_target.pt", weights_only=True)
    test_images = torch.load(f"{project_dir}/data/processed/cards-dataset/test_images.pt", weights_only=True)
    test_target = torch.load(f"{project_dir}/data/processed/cards-dataset/test_target.pt", weights_only=True)

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    valid_set = torch.utils.data.TensorDataset(valid_images, valid_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, valid_set, test_set


if __name__ == "__main__":
    fetch_kaggle()
    preprocess_data()
    # typer.run(fetch_kaggle)

    # typer.run(preprocess_data) # python .\src\project\data.py .\data\raw\corruptmnist .\data\processed\corruptmnist
