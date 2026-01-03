import torch
from torch.utils.data import Dataset
import random
from typing import Optional, List, Tuple
import os


def dataloader(dir_path: str):
    sequences = {
        "train": {
            "positive": [],
            "negative": []
        },
        "test": {
            "positive": [],
            "negative": []
        }
    }

    for entry in os.scandir(dir_path):
        print(entry.name)
        subdir_path = os.path.join(dir_path, entry.name)
        for subentry in os.scandir(subdir_path):
            print(subentry.name)
            subentry_path = os.path.join(subdir_path, subentry.name)
            for file in os.listdir(subentry_path):
                with open(os.path.join(subentry_path, file), "r") as f:
                    sequences[entry.name][subentry.name].append(f.read())

    return sequences


def main():
    sequences = dataloader("data/human_nontata_promoters/")
    print(sequences["train"]["positive"][0])
    print(len(sequences["train"]["positive"]))
    #print(sequences["train"]["negative"][0])
    #print(len(sequences["train"]["negative"]))
    #print(sequences["test"]["positive"][0])
    #print(len(sequences["test"]["positive"]))
    #print(sequences["test"]["negative"][0])
    #print(len(sequences["test"]["negative"]))

if __name__ == "__main__":
    main()