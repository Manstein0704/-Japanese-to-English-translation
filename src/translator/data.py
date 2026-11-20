"""JESC dataset loading and preprocessing."""

from pathlib import Path
from tokenizers import Tokenizer
from datasets import load_dataset
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def load_jesc_datasets():
    dataset = load_dataset("nntsuzu/JESC", split="train")
    split = dataset.train_test_split(test_size=8000, seed=2026)
    eval_split = split["test"].train_test_split(test_size=4000, seed=2026)

    train_ds = split["train"]
    eval_ds = eval_split["train"]
    test_ds = eval_split["test"]


    return train_ds, eval_ds, test_ds



def get_batch(dataset, batch_size, pad_id=0):
    src_ids, tgt_ids = dataset

    indicies = torch.randint(0, len(src_ids), (batch_size,))
    src_batch = [torch.tensor(src_ids[i], dtype=torch.long)
                 for i in indicies.tolist()]
    tgt_batch = [torch.tensor(tgt_ids[i], dtype=torch.long)
                 for i in indicies.tolist()]

    src = pad_sequence(src_batch, batch_first=True, padding_value=pad_id)
    tgt = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_id)

    decoder_input = tgt[:,:-1]
    labels = tgt[:, 1:]

    src_padding_mask = (src==pad_id)
    tgt_padding_mask = (tgt==pad_id)


    return src, decoder_input, labels, src_padding_mask, tgt_padding_mask



def tokenize_dataset(dataset, tokenizer:Tokenizer):
    src_ids = []
    tgt_ids = []

    for example in dataset:
        ja = example["translation"]["ja"]
        en = example["translation"]["en"]

        src_ids.append(tokenizer.encode(ja).ids)
        tgt_ids.append(tokenizer.encode(en).ids)

    return src_ids, tgt_ids



def tokenize_and_save(dataset, tokenizer:Tokenizer, output_path, batch_size=1000):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if tokenizer.get_vocab_size() > 2**16:
        raise ValueError("Vocabulary too large for uint16")

    total_tokens = 0
    with open(output_path, "wb") as f:
        for start in range(0, len(dataset), batch_size):
            

