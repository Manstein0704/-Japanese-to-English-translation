"""JESC dataset loading and preprocessing."""

from pathlib import Path
from tokenizers import Tokenizer
from datasets import load_dataset
import numpy as np
import torch


def load_jesc_datasets():
    dataset = load_dataset("nntsuzu/JESC", split="train")
    split = dataset.train_test_split(test_size=8000, seed=2026)
    eval_split = split["test"].train_test_split(test_size=4000, seed=2026)

    train_ds = split["train"]
    eval_ds = eval_split["train"]
    test_ds = eval_split["test"]


    return train_ds, eval_ds, test_ds



def get_batch(dataset, tokenizer, batch_size=32):
    raise NotImplementedError





if __name__ == "__main__":
    train_ds, eval_ds, test_ds = load_jesc_datasets()
    print(f"Train:{len(train_ds)}")
    print(f"Validation:{len(eval_ds)}")
    print(f"Test:{len(test_ds)}")

    print(train_ds[0])

